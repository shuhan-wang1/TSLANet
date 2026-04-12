"""
test.py — ProbabilisticTSLANet Testing & Evaluation Script
===========================================================
Loads a saved model, runs inference (deterministic / Gaussian / MC Dropout),
computes metrics (MSE, MAE, RMSE, NLL, CRPS, calibration, sharpness),
and generates visualisation plots.

Usage:
  python test.py --model_dir saved_models/<run_description>
  python test.py --model_dir saved_models/<run_description> --mc_samples 100

All helper modules (model, data loading, inference, metrics, visualisation)
are embedded in this single file.

GenAI Statement: Claude was used in an assistive role during development.
We first studied the original TSLANet paper (Eldele et al., ICML 2024) and
its reference implementation, then independently designed the algorithms to
extend TSLANet into a probabilistic forecasting model (Gaussian likelihood
heads, MC Dropout for epistemic uncertainty, calibration metrics, etc.).
Claude was used only to assist with debugging and fixing errors encountered
during this extension, as well as code restructuring and documentation.
All core algorithmic design decisions and logic were made by us and
verified manually.
"""

import argparse
import datetime
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')


# ===========================================================================
# Section 1: Utilities
# ===========================================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def random_masking_3D(xb, mask_ratio):
    bs, L, D = xb.shape
    x = xb.clone()
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(bs, L, device=xb.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, x_kept, mask, ids_restore


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        tensor.normal_(mean, std)
        while True:
            cond = (tensor < mean - 2 * std) | (tensor > mean + 2 * std)
            if not cond.any():
                break
            tensor[cond] = tensor[cond].normal_(mean, std)
    return tensor


# ===========================================================================
# Section 2: Time Features (from GluonTS, Apache 2.0 License)
# ===========================================================================

class TimeFeature:
    def __init__(self): pass
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray: pass
    def __repr__(self): return self.__class__.__name__ + "()"

class SecondOfMinute(TimeFeature):
    def __call__(self, index): return index.second / 59.0 - 0.5
class MinuteOfHour(TimeFeature):
    def __call__(self, index): return index.minute / 59.0 - 0.5
class HourOfDay(TimeFeature):
    def __call__(self, index): return index.hour / 23.0 - 0.5
class DayOfWeek(TimeFeature):
    def __call__(self, index): return index.dayofweek / 6.0 - 0.5
class DayOfMonth(TimeFeature):
    def __call__(self, index): return (index.day - 1) / 30.0 - 0.5
class DayOfYear(TimeFeature):
    def __call__(self, index): return (index.dayofyear - 1) / 365.0 - 0.5
class MonthOfYear(TimeFeature):
    def __call__(self, index): return (index.month - 1) / 11.0 - 0.5
class WeekOfYear(TimeFeature):
    def __call__(self, index): return (index.isocalendar().week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str):
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
    }
    offset = to_offset(freq_str)
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]
    raise RuntimeError(f"Unsupported frequency {freq_str}")

def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


# ===========================================================================
# Section 3: Data Loading
# ===========================================================================

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std == 0] = 1.0
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return data * self.std + self.mean


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 24 * 4 * 4; self.label_len = 24 * 4; self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]; self.label_len = size[1]; self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features; self.target = target; self.scale = scale
        self.timeenc = timeenc; self.freq = freq
        self.root_path = root_path; self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s = [0, 12*30*24 - self.seq_len, 12*30*24 + 4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
        border1 = border1s[self.set_type]; border2 = border2s[self.set_type]
        if self.features in ('M', 'MS'):
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index; s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len; r_end = r_begin + self.label_len + self.pred_len
        return (self.data_x[s_begin:s_end], self.data_y[r_begin:r_end],
                self.data_stamp[s_begin:s_end], self.data_stamp[r_begin:r_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        if size is None:
            self.seq_len = 24*4*4; self.label_len = 24*4; self.pred_len = 24*4
        else:
            self.seq_len = size[0]; self.label_len = size[1]; self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features; self.target = target; self.scale = scale
        self.timeenc = timeenc; self.freq = freq
        self.root_path = root_path; self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4 + 4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4]
        border1 = border1s[self.set_type]; border2 = border2s[self.set_type]
        if self.features in ('M', 'MS'):
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index; s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len; r_end = r_begin + self.label_len + self.pred_len
        return (self.data_x[s_begin:s_end], self.data_y[r_begin:r_end],
                self.data_stamp[s_begin:s_end], self.data_stamp[r_begin:r_end])
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        if size is None:
            self.seq_len = 24*4*4; self.label_len = 24*4; self.pred_len = 24*4
        else:
            self.seq_len = size[0]; self.label_len = size[1]; self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features; self.target = target; self.scale = scale
        self.timeenc = timeenc; self.freq = freq
        self.root_path = root_path; self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]; border2 = border2s[self.set_type]
        if self.features in ('M', 'MS'):
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index; s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len; r_end = r_begin + self.label_len + self.pred_len
        return (self.data_x[s_begin:s_end], self.data_y[r_begin:r_end],
                self.data_stamp[s_begin:s_end], self.data_stamp[r_begin:r_end])
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


DATA_DICT = {
    'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

def data_provider(args, flag):
    Data = DATA_DICT[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = (flag != 'test')
    data_set = Data(
        root_path=args.root_path, data_path=args.data_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, target=args.target, timeenc=timeenc, freq=args.freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(data_set, batch_size=args.batch_size,
                             shuffle=shuffle_flag, drop_last=True)
    return data_set, data_loader


# ===========================================================================
# Section 4: Model Architecture — ProbabilisticTSLANet
# ===========================================================================

class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x); x1_1 = self.act(x1); x1_2 = self.drop(x1_1)
        x2 = self.conv2(x); x2_1 = self.act(x2); x2_2 = self.drop(x2_1)
        out1 = x1 * x2_2; out2 = x2 * x1_2
        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(B, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        return adaptive_mask.unsqueeze(-1)

    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight
        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted += x_masked * weight_high
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        return x.to(dtype).view(B, N, C)


class TSLANet_layer(nn.Module):
    def __init__(self, dim, args, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.args = args
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim, adaptive_filter=args.adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.icb = ICB(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        if self.args.ICB and self.args.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.args.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.args.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class ProbabilisticTSLANet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_size = args.patch_size
        self.stride = self.patch_size // 2
        self.num_patches = int((args.seq_len - self.patch_size) / self.stride + 1)
        self.input_layer = nn.Linear(self.patch_size, args.emb_dim)
        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, args=args, drop=args.dropout, drop_path=dpr[i])
            for i in range(args.depth)])
        backbone_out_dim = args.emb_dim * self.num_patches
        if args.probabilistic:
            self.mu_head = nn.Linear(backbone_out_dim, args.pred_len)
            self.log_var_head = nn.Linear(backbone_out_dim, args.pred_len)
            nn.init.zeros_(self.log_var_head.weight)
            nn.init.constant_(self.log_var_head.bias, -2.0)
        else:
            self.out_layer = nn.Linear(backbone_out_dim, args.pred_len)

    def pretrain(self, x_in):
        x = x_in.permute(0, 2, 1)
        x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        B, M = x_patched.shape[0], x_patched.shape[1]
        x_patched = x_patched.reshape(B * M, self.num_patches, self.patch_size)
        xb_mask, _, self.mask, _ = random_masking_3D(x_patched, mask_ratio=self.args.mask_ratio)
        self.mask = self.mask.bool()
        xb_mask = self.input_layer(xb_mask)
        for tsla_blk in self.tsla_blocks:
            xb_mask = tsla_blk(xb_mask)
        return xb_mask, self.input_layer(x_patched)

    def forward(self, x):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = x.reshape(B * M, self.num_patches, self.patch_size)
        x = self.input_layer(x)
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)
        features = x.reshape(B * M, -1)

        if self.args.probabilistic:
            mu_norm = self.mu_head(features)
            log_var_norm = torch.clamp(self.log_var_head(features), -10, 10)
            mu_norm = mu_norm.reshape(B, M, -1).permute(0, 2, 1)
            log_var_norm = log_var_norm.reshape(B, M, -1).permute(0, 2, 1)
            mu = mu_norm * stdev + means
            log_var = log_var_norm + 2 * torch.log(stdev + 1e-5)
            return mu, log_var
        else:
            outputs = self.out_layer(features)
            outputs = outputs.reshape(B, M, -1).permute(0, 2, 1)
            return outputs * stdev + means


# ===========================================================================
# Section 5: Inference Functions
# ===========================================================================

def deterministic_predict(model, dataloader, pred_len):
    """A1: Deterministic model, single forward pass."""
    model.eval()
    all_mu, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y, _, _ in dataloader:
            batch_x = batch_x.float()
            batch_y = batch_y[:, -pred_len:, :].float()
            outputs = model(batch_x)[:, -pred_len:, :]
            all_mu.append(outputs)
            all_targets.append(batch_y)
    mu_mean = torch.cat(all_mu, dim=0)
    targets = torch.cat(all_targets, dim=0)
    zeros = torch.zeros_like(mu_mean)
    return {'mu_mean': mu_mean, 'targets': targets,
            'epistemic_var': zeros, 'aleatoric_var': zeros, 'total_var': zeros}


def gaussian_predict(model, dataloader, pred_len):
    """A3: Gaussian head, single forward pass, aleatoric only."""
    model.eval()
    all_mu, all_log_var, all_targets = [], [], []
    with torch.no_grad():
        for batch_x, batch_y, _, _ in dataloader:
            batch_x = batch_x.float()
            batch_y = batch_y[:, -pred_len:, :].float()
            mu, log_var = model(batch_x)
            all_mu.append(mu[:, -pred_len:, :])
            all_log_var.append(log_var[:, -pred_len:, :])
            all_targets.append(batch_y)
    mu_mean = torch.cat(all_mu, dim=0)
    log_var = torch.cat(all_log_var, dim=0)
    targets = torch.cat(all_targets, dim=0)
    aleatoric_var = torch.exp(log_var)
    zeros = torch.zeros_like(mu_mean)
    return {'mu_mean': mu_mean, 'targets': targets,
            'epistemic_var': zeros, 'aleatoric_var': aleatoric_var, 'total_var': aleatoric_var}


def mc_dropout_predict(model, dataloader, pred_len, num_samples=50):
    """A2 & A4: K stochastic forward passes with dropout enabled."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    is_probabilistic = hasattr(model, 'mu_head')
    all_targets = None
    all_mu_samples, all_log_var_samples = [], []

    for k in range(num_samples):
        batch_mus, batch_log_vars, batch_targets = [], [], []
        with torch.no_grad():
            for batch_x, batch_y, _, _ in dataloader:
                batch_x = batch_x.float()
                batch_y = batch_y[:, -pred_len:, :].float()
                if is_probabilistic:
                    mu, log_var = model(batch_x)
                    batch_mus.append(mu[:, -pred_len:, :])
                    batch_log_vars.append(log_var[:, -pred_len:, :])
                else:
                    outputs = model(batch_x)[:, -pred_len:, :]
                    batch_mus.append(outputs)
                if all_targets is None:
                    batch_targets.append(batch_y)

        all_mu_samples.append(torch.cat(batch_mus, dim=0))
        if is_probabilistic:
            all_log_var_samples.append(torch.cat(batch_log_vars, dim=0))
        if all_targets is None:
            all_targets = torch.cat(batch_targets, dim=0)

    mu_stack = torch.stack(all_mu_samples, dim=0)
    mu_mean = mu_stack.mean(dim=0)
    epistemic_var = mu_stack.var(dim=0)

    if is_probabilistic:
        log_var_stack = torch.stack(all_log_var_samples, dim=0)
        aleatoric_var = torch.exp(log_var_stack).mean(dim=0)
    else:
        aleatoric_var = torch.zeros_like(mu_mean)

    return {'mu_mean': mu_mean, 'targets': all_targets,
            'epistemic_var': epistemic_var, 'aleatoric_var': aleatoric_var,
            'total_var': epistemic_var + aleatoric_var}


# ===========================================================================
# Section 6: Metrics
# ===========================================================================

def compute_nll(mu, log_var, target):
    """Gaussian NLL with log(2*pi) constant for reporting."""
    precision = torch.exp(-log_var)
    nll = 0.5 * (log_var + (target - mu) ** 2 * precision + math.log(2 * math.pi))
    return nll.mean().item()


def compute_crps(mu, log_var, target):
    """Analytic Gaussian CRPS."""
    sigma = torch.exp(0.5 * log_var)
    z = (target - mu) / (sigma + 1e-6)
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return crps.mean().item()


def compute_calibration(mu, total_var, target, quantiles=None):
    """Compute observed coverage for each nominal coverage level."""
    if quantiles is None:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    sigma = torch.sqrt(total_var + 1e-8)
    results = {}
    for q in quantiles:
        z = torch.erfinv(torch.tensor(q, dtype=torch.float64)).item() * math.sqrt(2)
        lower = mu - z * sigma
        upper = mu + z * sigma
        covered = ((target >= lower) & (target <= upper)).float().mean().item()
        results[q] = covered
    return results


def compute_sharpness(total_var, confidence=0.9):
    """Average width of prediction interval at given confidence level."""
    z = torch.erfinv(torch.tensor(confidence, dtype=torch.float64)).item() * math.sqrt(2)
    sigma = torch.sqrt(total_var + 1e-8)
    width = 2.0 * z * sigma
    return width.mean().item()


def compute_all_metrics(results_dict):
    """Compute all probabilistic and point metrics from inference results."""
    mu = results_dict['mu_mean']
    target = results_dict['targets']
    total_var = results_dict['total_var']
    epistemic_var = results_dict['epistemic_var']
    aleatoric_var = results_dict['aleatoric_var']

    metrics = {}
    metrics['MSE'] = F.mse_loss(mu, target).item()
    metrics['MAE'] = F.l1_loss(mu, target).item()
    metrics['RMSE'] = math.sqrt(metrics['MSE'])

    has_uncertainty = total_var.sum().item() > 0
    if has_uncertainty:
        log_total_var = torch.log(total_var + 1e-8)
        metrics['NLL'] = compute_nll(mu, log_total_var, target)
        metrics['CRPS'] = compute_crps(mu, log_total_var, target)
        metrics['Sharpness_90'] = compute_sharpness(total_var, 0.9)
        metrics['Sharpness_50'] = compute_sharpness(total_var, 0.5)

        calibration = compute_calibration(mu, total_var, target)
        for nom_cov, obs_cov in calibration.items():
            metrics[f'Cal_{nom_cov}'] = obs_cov
        cal_error = np.mean([abs(nom - obs) for nom, obs in calibration.items()])
        metrics['Calibration_Error'] = float(cal_error)

        metrics['Mean_Epistemic_Var'] = epistemic_var.mean().item()
        metrics['Mean_Aleatoric_Var'] = aleatoric_var.mean().item()
        total_mean = total_var.mean().item()
        metrics['Epistemic_Fraction'] = epistemic_var.mean().item() / (total_mean + 1e-8)

    return metrics


# ===========================================================================
# Section 7: Visualisation
# ===========================================================================

def plot_prediction_intervals(mu, total_var, target, variable_idx=0, sample_idx=0,
                              save_path='prediction_intervals.pdf'):
    mu_s = mu[sample_idx, :, variable_idx].numpy()
    sigma_s = np.sqrt(total_var[sample_idx, :, variable_idx].numpy())
    target_s = target[sample_idx, :, variable_idx].numpy()
    T = len(mu_s)
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 5))
    for coverage, alpha, label in [(0.95, 0.15, '95%'), (0.9, 0.25, '90%'), (0.5, 0.4, '50%')]:
        z = float(torch.erfinv(torch.tensor(coverage, dtype=torch.float64)).item()) * math.sqrt(2)
        lower = mu_s - z * sigma_s
        upper = mu_s + z * sigma_s
        ax.fill_between(t, lower, upper, alpha=alpha, color='steelblue', label=f'{label} PI')
    ax.plot(t, mu_s, color='steelblue', linewidth=1.5, label='Predicted mean')
    ax.plot(t, target_s, color='darkorange', linewidth=1.5, linestyle='--', label='Ground truth')
    ax.set_xlabel('Forecast Horizon (timestep)')
    ax.set_ylabel('Value')
    ax.set_title(f'Prediction Intervals (Variable {variable_idx}, Sample {sample_idx})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction interval plot to {save_path}")


def plot_calibration(mu, total_var, target, save_path='calibration.pdf'):
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    cal = compute_calibration(mu, total_var, target, quantiles=quantiles)
    nominal = list(cal.keys())
    observed = list(cal.values())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.plot(nominal, observed, 'o-', color='steelblue', linewidth=2, markersize=8, label='Model')
    ax.fill_between(nominal, nominal, observed, alpha=0.15, color='steelblue')
    ax.set_xlabel('Nominal Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title('Calibration Plot')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration plot to {save_path}")


def plot_uncertainty_decomposition(epistemic_var, aleatoric_var, variable_idx=0,
                                   save_path='uncertainty_decomposition.pdf'):
    epi = epistemic_var[:, :, variable_idx].mean(dim=0).numpy()
    ale = aleatoric_var[:, :, variable_idx].mean(dim=0).numpy()
    T = len(epi)
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(t, 0, ale, alpha=0.6, color='#2196F3', label='Aleatoric')
    ax.fill_between(t, ale, ale + epi, alpha=0.6, color='#FF5722', label='Epistemic')
    ax.set_xlabel('Forecast Horizon (timestep)')
    ax.set_ylabel('Variance')
    ax.set_title(f'Uncertainty Decomposition (Variable {variable_idx})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved uncertainty decomposition plot to {save_path}")


def plot_uncertainty_heatmap(epistemic_var, aleatoric_var, target, mu,
                             sample_idx=0, max_variables=None,
                             variable_names=None, save_path='uncertainty_heatmap.pdf'):
    M = epistemic_var.shape[2]
    if max_variables is not None:
        M = min(M, max_variables)

    epi = epistemic_var[sample_idx, :, :M].numpy()
    ale = aleatoric_var[sample_idx, :, :M].numpy()
    total = epi + ale
    mu_s = mu[sample_idx, :, :M].numpy()
    tgt_s = target[sample_idx, :, :M].numpy()
    T = epi.shape[0]

    if variable_names is None:
        variable_names = [f'Var {i}' for i in range(M)]
    else:
        variable_names = variable_names[:M]

    cmap_ale = LinearSegmentedColormap.from_list('aleatoric', ['#FFFFFF', '#BBDEFB', '#1565C0', '#0D47A1'], N=256)
    cmap_epi = LinearSegmentedColormap.from_list('epistemic', ['#FFFFFF', '#FFCCBC', '#E64A19', '#BF360C'], N=256)
    cmap_total = LinearSegmentedColormap.from_list('total', ['#FFFFFF', '#E1BEE7', '#7B1FA2', '#4A148C'], N=256)
    cmap_err = LinearSegmentedColormap.from_list('error', ['#E8F5E9', '#FFF9C4', '#FFAB91', '#C62828'], N=256)

    fig = plt.figure(figsize=(max(14, T * 0.12), 3 + M * 0.55 * 3))
    gs = GridSpec(4, 2, width_ratios=[50, 1], hspace=0.35, wspace=0.05)

    rows = [
        (0, np.abs(mu_s - tgt_s).T, cmap_err, 'Absolute Prediction Error  |y - $\\hat{y}$|', 'black'),
        (1, ale.T, cmap_ale, 'Aleatoric Uncertainty (Data Noise)', '#1565C0'),
        (2, epi.T, cmap_epi, 'Epistemic Uncertainty (Model Uncertainty)', '#E64A19'),
        (3, total.T, cmap_total, 'Total Uncertainty (Aleatoric + Epistemic)', '#7B1FA2'),
    ]

    for row_idx, data_arr, cmap, title, color in rows:
        ax = fig.add_subplot(gs[row_idx, 0])
        ax_cb = fig.add_subplot(gs[row_idx, 1])
        im = ax.imshow(data_arr, aspect='auto', cmap=cmap, interpolation='nearest')
        ax.set_yticks(np.arange(M))
        ax.set_yticklabels(variable_names, fontsize=8)
        ax.set_xlabel('Forecast Horizon (timestep)', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        cb = fig.colorbar(im, ax=ax_cb, fraction=0.9, pad=0.0)
        cb.ax.tick_params(labelsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved uncertainty heatmap to {save_path}")


def generate_all_plots(results_dict, output_dir, prefix=''):
    os.makedirs(output_dir, exist_ok=True)
    mu = results_dict['mu_mean']
    total_var = results_dict['total_var']
    target = results_dict['targets']
    epistemic_var = results_dict['epistemic_var']
    aleatoric_var = results_dict['aleatoric_var']

    for i in range(min(3, mu.shape[0])):
        plot_prediction_intervals(mu, total_var, target, variable_idx=0, sample_idx=i,
            save_path=os.path.join(output_dir, f'{prefix}pred_intervals_sample{i}.pdf'))

    plot_calibration(mu, total_var, target,
        save_path=os.path.join(output_dir, f'{prefix}calibration.pdf'))

    for v in range(min(3, mu.shape[2])):
        plot_uncertainty_decomposition(epistemic_var, aleatoric_var, variable_idx=v,
            save_path=os.path.join(output_dir, f'{prefix}uncertainty_decomp_var{v}.pdf'))

    plot_uncertainty_heatmap(epistemic_var, aleatoric_var, target, mu, sample_idx=0,
        max_variables=min(21, mu.shape[2]),
        save_path=os.path.join(output_dir, f'{prefix}uncertainty_heatmap.pdf'))


# ===========================================================================
# Section 8: Main — Test Entry Point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Test ProbabilisticTSLANet')
    parser.add_argument('--model_dir', type=str, default='saved_models/full_model_pl96',
                        help='Path to saved model directory (default: saved_models/full_model_pl96)')
    parser.add_argument('--mc_samples', type=int, default=None, help='Override MC samples')
    parser.add_argument('--root_path', type=str, default=None, help='Override data root path')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    return parser.parse_args()


def load_config(model_dir):
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return argparse.Namespace(**config)


def main():
    test_args = parse_args()
    args = load_config(test_args.model_dir)

    # Apply overrides
    if test_args.mc_samples is not None:
        args.mc_samples = test_args.mc_samples
    if test_args.root_path is not None:
        args.root_path = test_args.root_path
    if test_args.batch_size is not None:
        args.batch_size = test_args.batch_size

    # Load data
    test_data, test_loader = data_provider(args, 'test')

    # Load model
    model = ProbabilisticTSLANet(args)
    model.load_state_dict(torch.load(
        os.path.join(test_args.model_dir, 'model_weights.pt'),
        map_location='cpu'))

    # Run inference (auto-detect mode from config)
    if args.mc_dropout:
        print(f"Running MC Dropout inference ({args.mc_samples} samples)...")
        results = mc_dropout_predict(model, test_loader, args.pred_len,
                                     num_samples=args.mc_samples)
    elif args.probabilistic:
        print("Running Gaussian inference (aleatoric only)...")
        results = gaussian_predict(model, test_loader, args.pred_len)
    else:
        print("Running deterministic inference...")
        results = deterministic_predict(model, test_loader, args.pred_len)

    # Compute metrics
    metrics = compute_all_metrics(results)

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Save results
    output_dir = os.path.join(test_args.model_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    json_metrics = {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(json_metrics, f, indent=2)

    # Generate plots (only if model has uncertainty)
    if args.probabilistic or args.mc_dropout:
        print("\nGenerating plots...")
        generate_all_plots(results, os.path.join(output_dir, 'plots'), prefix='')

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()