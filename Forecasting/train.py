"""
train.py — ProbabilisticTSLANet Training Script
================================================
Trains a ProbabilisticTSLANet model on sequential data with optional:
  - Self-supervised pretraining (masked patch reconstruction)
  - Gaussian NLL loss for aleatoric uncertainty
  - MC Dropout for epistemic uncertainty

Usage:
  python train.py --probabilistic True --mc_dropout True --pred_len 96
  python train.py --probabilistic False --mc_dropout False --pred_len 336

All helper modules (model, data loading, losses, utilities) are embedded
in this single file.

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
from torch.utils.data import Dataset, DataLoader
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import List

warnings.filterwarnings('ignore')


# ===========================================================================
# Section 1: Utilities
# ===========================================================================

def str2bool(v):
    """Parse boolean values from command-line arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def random_masking_3D(xb, mask_ratio):
    """Random masking for self-supervised pretraining.

    Args:
        xb: input tensor of shape (bs, num_patch, dim)
        mask_ratio: fraction of patches to mask

    Returns:
        x_masked, x_kept, mask, ids_restore
    """
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

    x_masked = torch.gather(x_, dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, x_kept, mask, ids_restore


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Stochastic depth drop path (from timm)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Stochastic depth module (from timm)."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def trunc_normal_(tensor, mean=0., std=1.):
    """Truncated normal initialization (from timm)."""
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
    def __init__(self):
        pass
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass
    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
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
    """Minimal replacement for sklearn StandardScaler using numpy."""
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
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
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
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
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
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
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
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# Data provider factory
DATA_DICT = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = DATA_DICT[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = (flag != 'test')

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=True)

    return data_set, data_loader


# ===========================================================================
# Section 4: Model Architecture — ProbabilisticTSLANet
# ===========================================================================

class ICB(nn.Module):
    """Interacting Convolutional Block: two parallel 1D convolutions with
    multiplicative gating, capturing local patterns and cross-feature interactions."""
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)
        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)
        out1 = x1 * x2_2
        out2 = x2 * x1_2
        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class Adaptive_Spectral_Block(nn.Module):
    """Applies FFT, learns frequency-domain weights, and uses an energy-based
    adaptive mask to separate low-frequency trends from high-frequency noise."""
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
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

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
            x_weighted2 = x_masked * weight_high
            x_weighted += x_weighted2

        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)
        return x


class TSLANet_layer(nn.Module):
    """Single TSLANet block: LayerNorm -> ASB -> LayerNorm -> ICB with residual."""
    def __init__(self, dim, args, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.args = args
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim, adaptive_filter=args.adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        if self.args.ICB and self.args.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.args.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.args.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class ProbabilisticTSLANet(nn.Module):
    """TSLANet extended with heteroscedastic Gaussian output heads for
    probabilistic forecasting and uncertainty quantification.

    Supports four configurations:
      A1: deterministic (MSE loss, no dropout at test time)
      A2: epistemic only (MSE loss + MC Dropout at test time)
      A3: aleatoric only (Gaussian NLL loss, no MC Dropout)
      A4: full uncertainty (Gaussian NLL + MC Dropout)
    """
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
            for i in range(args.depth)]
        )

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

        # Instance normalization (RevIN-style)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        # Patching
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = x.reshape(B * M, self.num_patches, self.patch_size)

        # Embedding + TSLANet blocks
        x = self.input_layer(x)
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        # Flatten
        features = x.reshape(B * M, -1)

        if self.args.probabilistic:
            mu_norm = self.mu_head(features)
            log_var_norm = self.log_var_head(features)
            log_var_norm = torch.clamp(log_var_norm, -10, 10)

            mu_norm = mu_norm.reshape(B, M, -1).permute(0, 2, 1)
            log_var_norm = log_var_norm.reshape(B, M, -1).permute(0, 2, 1)

            # Denormalize
            mu = mu_norm * stdev + means
            log_var = log_var_norm + 2 * torch.log(stdev + 1e-5)

            return mu, log_var
        else:
            outputs = self.out_layer(features)
            outputs = outputs.reshape(B, M, -1).permute(0, 2, 1)
            outputs = outputs * stdev + means
            return outputs


# ===========================================================================
# Section 5: Loss Function
# ===========================================================================

class GaussianNLLLoss(nn.Module):
    """Gaussian negative log-likelihood loss.

    NLL = 0.5 * mean(log_var + (target - mu)^2 / exp(log_var))
    Constant term omitted during training (does not affect gradients).
    """
    def __init__(self):
        super().__init__()

    def forward(self, mu, log_var, target):
        precision = torch.exp(-log_var)
        loss = 0.5 * (log_var + (target - mu) ** 2 * precision)
        return loss.mean()


# ===========================================================================
# Section 6: Training Logic
# ===========================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train ProbabilisticTSLANet')

    # Data
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='data/weather')
    parser.add_argument('--data_path', type=str, default='weather.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    # Model
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--mask_ratio', type=float, default=0.4)

    # TSLANet components
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--pretrain_epochs', type=int, default=5)
    parser.add_argument('--train_epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)

    # Pretraining
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True)

    # Probabilistic
    parser.add_argument('--probabilistic', type=str2bool, default=True)
    parser.add_argument('--mc_dropout', type=str2bool, default=True)
    parser.add_argument('--mc_samples', type=int, default=50)
    parser.add_argument('--aux_mse_weight', type=float, default=0.0,
                        help='A5: weight for auxiliary MSE loss added to NLL (e.g. 0.3)')
    parser.add_argument('--two_stage', type=str2bool, default=False,
                        help='A6: two-stage training — stage1 MSE on backbone+mu_head, stage2 NLL on log_var_head only')

    # Output
    parser.add_argument('--save_dir', type=str, default='saved_models')

    return parser.parse_args()


def make_save_dir(args):
    data_name = args.data_path.split('.')[0]
    if args.two_stage:
        prob_tag = 'prob_twostage'
    elif args.probabilistic and args.aux_mse_weight > 0:
        prob_tag = f'prob_auxmse{args.aux_mse_weight}'
    elif args.probabilistic:
        prob_tag = 'prob'
    else:
        prob_tag = 'det'
    mc_tag = 'mc' if args.mc_dropout else 'nomc'
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    desc = f"{data_name}_{prob_tag}_{mc_tag}_pl{args.pred_len}_{timestamp}"
    save_dir = os.path.join(args.save_dir, desc)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_config(args, save_dir):
    config = vars(args).copy()
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def evaluate_pretrain(model, val_loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_x, batch_y, _, _ in val_loader:
            batch_x = batch_x.float()
            preds, target = model.pretrain(batch_x)
            loss = (preds - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * model.mask).sum() / model.mask.sum()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, val_loader, args):
    model.eval()
    total_mse = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_x, batch_y, _, _ in val_loader:
            batch_x = batch_x.float()
            batch_y = batch_y[:, -args.pred_len:, :].float()

            if args.probabilistic:
                mu, log_var = model(batch_x)
                mu = mu[:, -args.pred_len:, :]
                mse = ((mu - batch_y) ** 2).mean()
            else:
                outputs = model(batch_x)
                outputs = outputs[:, -args.pred_len:, :]
                mse = ((outputs - batch_y) ** 2).mean()

            total_mse += mse.item()
            n_batches += 1
    return total_mse / max(n_batches, 1)


def pretrain_phase(model, train_loader, val_loader, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(args.pretrain_epochs):
        model.train()
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float()
            preds, target = model.pretrain(batch_x)
            loss = (preds - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * model.mask).sum() / model.mask.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = evaluate_pretrain(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs} | val_loss: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)


def train_phase(model, train_loader, val_loader, args):
    criterion = GaussianNLLLoss() if args.probabilistic else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val_mse = float('inf')
    best_state = None

    for epoch in range(args.train_epochs):
        model.train()
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float()
            batch_y = batch_y[:, -args.pred_len:, :].float()

            if args.probabilistic:
                mu, log_var = model(batch_x)
                mu = mu[:, -args.pred_len:, :]
                log_var = log_var[:, -args.pred_len:, :]
                loss = criterion(mu, log_var, batch_y)
                if args.aux_mse_weight > 0:
                    loss = loss + args.aux_mse_weight * nn.functional.mse_loss(mu, batch_y)
            else:
                outputs = model(batch_x)
                outputs = outputs[:, -args.pred_len:, :]
                loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

        val_mse = evaluate(model, val_loader, args)
        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1}/{args.train_epochs} | val_mse: {val_mse:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state


def train_phase_two_stage(model, train_loader, val_loader, args):
    """A6: Two-stage training.

    Stage 1 — train backbone + mu_head with MSE so point predictions converge first.
    Stage 2 — freeze backbone + mu_head, train only log_var_head with Gaussian NLL
               so uncertainty is learned without disturbing mu accuracy.
    """
    mse_criterion = nn.MSELoss()
    nll_criterion = GaussianNLLLoss()

    # ── Stage 1: MSE on backbone + mu_head ──────────────────────────────────
    print("=== A6 Stage 1: MSE training (backbone + mu_head) ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    best_val_mse = float('inf')
    best_state = None

    for epoch in range(args.train_epochs):
        model.train()
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float()
            batch_y = batch_y[:, -args.pred_len:, :].float()
            mu, _ = model(batch_x)
            mu = mu[:, -args.pred_len:, :]
            loss = mse_criterion(mu, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

        val_mse = evaluate(model, val_loader, args)
        scheduler.step(val_mse)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  Stage1 Epoch {epoch+1}/{args.train_epochs} | val_mse: {val_mse:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Stage 2: NLL on log_var_head only (backbone + mu_head frozen) ───────
    print("=== A6 Stage 2: NLL training (log_var_head only) ===")
    for name, param in model.named_parameters():
        param.requires_grad = ('log_var_head' in name)

    optimizer2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.1, patience=3)
    best_val_mse2 = float('inf')
    best_state2 = None

    for epoch in range(args.train_epochs):
        model.train()
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float()
            batch_y = batch_y[:, -args.pred_len:, :].float()
            mu, log_var = model(batch_x)
            mu = mu[:, -args.pred_len:, :].detach()
            log_var = log_var[:, -args.pred_len:, :]
            loss = nll_criterion(mu, log_var, batch_y)
            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer2.step()

        val_mse = evaluate(model, val_loader, args)
        scheduler2.step(val_mse)
        if val_mse < best_val_mse2:
            best_val_mse2 = val_mse
            best_state2 = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  Stage2 Epoch {epoch+1}/{args.train_epochs} | val_mse: {val_mse:.6f}")

    if best_state2 is not None:
        model.load_state_dict(best_state2)

    # Restore all params to be trainable before saving
    for param in model.parameters():
        param.requires_grad = True

    return best_state2


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load data
    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')

    # Create model
    model = ProbabilisticTSLANet(args)

    # Phase 1: Self-supervised pretraining
    if args.load_from_pretrained:
        print("=== Phase 1: Self-supervised pretraining ===")
        pretrain_phase(model, train_loader, val_loader, args)

    # Phase 2: Supervised fine-tuning
    print("=== Phase 2: Supervised fine-tuning ===")
    if args.two_stage:
        best_state = train_phase_two_stage(model, train_loader, val_loader, args)
    else:
        best_state = train_phase(model, train_loader, val_loader, args)

    # Save
    save_dir = make_save_dir(args)
    if best_state is not None:
        torch.save(best_state, os.path.join(save_dir, 'model_weights.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pt'))
    save_config(args, save_dir)

    print(f"Model saved to: {save_dir}")


if __name__ == '__main__':
    main()