import torch
import torch.nn as nn

from utils import DropPath, trunc_normal_, random_masking_3D


class ICB(nn.Module):
    """Interacting Convolutional Block."""
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

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy and compute median
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)

        return x


class TSLANet_layer(nn.Module):
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
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.patch_size = args.patch_size
        self.stride = self.patch_size // 2
        self.num_patches = int((args.seq_len - self.patch_size) / self.stride + 1)

        # Layers/Networks
        self.input_layer = nn.Linear(self.patch_size, args.emb_dim)

        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, args=args, drop=args.dropout, drop_path=dpr[i])
            for i in range(args.depth)]
        )

        backbone_out_dim = args.emb_dim * self.num_patches

        # Output heads
        if args.probabilistic:
            self.mu_head = nn.Linear(backbone_out_dim, args.pred_len)
            self.log_var_head = nn.Linear(backbone_out_dim, args.pred_len)
            # Initialize log_var_head for conservative initial variance
            nn.init.zeros_(self.log_var_head.weight)
            nn.init.constant_(self.log_var_head.bias, -2.0)
        else:
            self.out_layer = nn.Linear(backbone_out_dim, args.pred_len)

    def pretrain(self, x_in):
        # x_in: (B, L, M)
        x = x_in.permute(0, 2, 1)  # (B, M, L)
        x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # x_patched: (B, M, num_patches, patch_size)
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
        x = x.permute(0, 2, 1)  # (B, M, L)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # x: (B, M, num_patches, patch_size)
        x = x.reshape(B * M, self.num_patches, self.patch_size)

        # Embedding + TSLANet blocks
        x = self.input_layer(x)
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        # Flatten
        features = x.reshape(B * M, -1)  # (B*M, backbone_out_dim)

        if self.args.probabilistic:
            mu_norm = self.mu_head(features)          # (B*M, pred_len)
            log_var_norm = self.log_var_head(features) # (B*M, pred_len)
            log_var_norm = torch.clamp(log_var_norm, -10, 10)

            # Reshape to (B, pred_len, M)
            mu_norm = mu_norm.reshape(B, M, -1).permute(0, 2, 1)
            log_var_norm = log_var_norm.reshape(B, M, -1).permute(0, 2, 1)

            # Denormalize
            mu = mu_norm * stdev + means
            log_var = log_var_norm + 2 * torch.log(stdev + 1e-5)

            return mu, log_var
        else:
            outputs = self.out_layer(features)  # (B*M, pred_len)
            outputs = outputs.reshape(B, M, -1).permute(0, 2, 1)  # (B, pred_len, M)

            # Denormalize
            outputs = outputs * stdev + means

            return outputs
