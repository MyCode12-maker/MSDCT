import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
from layers.Embed import DataEmbedding_wo_pos

class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, degree=0, stride=1, padding=0, dilation=1, groups=1, act=False,
                 bn=False, bias=False, dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
class FrequencyFeatureBlock(nn.Module):
    def __init__(self, seq_len, chnall, hidden_ratio=2, dropout=0.1):
        super().__init__()
        hidden = max(seq_len, seq_len * hidden_ratio)

        self.band_mixer = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, seq_len),
        )

        self.conv1 = BasicConv(chnall, chnall, kernel_size=3, groups=chnall)

    def forward(self, x):
        # x: [B, N, T_s]

        x_k = self.band_mixer(x)
        x_c = self.conv1(x_k.permute(0, 2, 1)).permute(0, 2, 1)

        return x_k + x_c

class MultiScaleFrequencyFusion(nn.Module):
    def __init__(self, n_scales, seq_len, dropout=0.1):
        super().__init__()
        self.scale_logits = nn.Parameter(torch.zeros(n_scales))

    def forward(self, scale_features):
        # scale_features: list of [B, N, T]
        stacked = torch.stack(scale_features, dim=-2)  # [B, N, S, T]
        scale_weight = torch.softmax(self.scale_logits, dim=0).view(1, 1, -1, 1)
        fused = (stacked * scale_weight).sum(dim=-2)
        fused = fused
        return fused

class FrequencyPredictionHead(nn.Module):
    def __init__(self, seq_len, pred_len, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(seq_len, seq_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * 2, pred_len),
        )

    def forward(self, x):
        # x: [B, N, T]
        return self.head(x)


class Model(nn.Module):
    """
    Multi-scale DCT frequency forecasting.

    Pipeline:
    input -> hierarchical multi-scale downsampling -> DCT -> frequency feature extraction
    -> upsample to the base spectral resolution -> multi-scale spectral fusion
    -> predict future spectrum -> IDCT -> time-domain forecast.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name in ["classification", "anomaly_detection", "imputation"]:
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.enc_in = configs.enc_in
        self.dropout = getattr(configs, "dropout", 0.1)

        self.down_sampling_window = max(1, getattr(configs, "down_sampling_window", 2))
        self.down_sampling_method = getattr(configs, "down_sampling_method", "avg").lower()
        requested_layers = max(0, getattr(configs, "down_sampling_layers", 2))

        self.scale_lens = [self.seq_len]
        cur_len = self.seq_len
        for _ in range(requested_layers):
            if cur_len <= 1:
                break
            if self.down_sampling_method in ["avg", "max"] and cur_len < self.down_sampling_window:
                break

            if self.down_sampling_method == "conv":
                next_len = (cur_len + self.down_sampling_window - 1) // self.down_sampling_window
            else:
                next_len = cur_len // self.down_sampling_window
            next_len = max(1, next_len)
            self.scale_lens.append(next_len)
            cur_len = next_len

        self.n_down_layers = len(self.scale_lens) - 1
        if self.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            self.down_pools = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=self.enc_in,
                        out_channels=self.enc_in,
                        kernel_size=3,
                        padding=padding,
                        stride=self.down_sampling_window,
                        padding_mode="circular",
                        bias=False,
                    )
                    for _ in range(self.n_down_layers)
                ]
            )
        else:
            self.down_pools = nn.ModuleList()

        hidden_ratio = getattr(configs, "freq_hidden_ratio", 2)
        self.scale_blocks = nn.ModuleList(
            [
                FrequencyFeatureBlock(
                    seq_len=scale_len,
                    chnall = self.enc_in,
                    hidden_ratio=hidden_ratio,
                    dropout=self.dropout,
                )
                for scale_len in self.scale_lens
            ]
        )

        self.fusion = MultiScaleFrequencyFusion(
            n_scales=len(self.scale_lens),
            seq_len=self.seq_len,
            dropout=self.dropout,
        )
        self.pred_head = FrequencyPredictionHead(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            dropout=self.dropout,
        )

        self.projection = None
        if self.task_name == "classification":
            self.projection = nn.Linear(configs.enc_in * self.pred_len, configs.num_class)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.d_models = 32

        self.enc_embedding = DataEmbedding_wo_pos(1, self.d_models, configs.embed, configs.freq,
                                                  configs.dropout)

        self.embeddingMixer = nn.Sequential(nn.Linear(self.d_models, self.d_models * 4),
                                            nn.GELU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(self.d_models * 4, 1))

        self.embeddingMixer2 = nn.Sequential(nn.Linear(self.seq_len, self.seq_len * 4),
                                             nn.GELU(),
                                             nn.Dropout(0.1),
                                             nn.Linear(self.seq_len * 4, self.pred_len))

    def _normalize(self, x):
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return x / stdev, means, stdev

    def _denormalize(self, x, means, stdev):
        return x * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

    def _resize_time(self, x, target_len):
        # x: [B, N, T_s]
        if x.size(-1) == target_len:
            return x
        return F.interpolate(x, size=target_len, mode="linear", align_corners=False)

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch
    def _multi_scale_downsample(self, x):
        # x: [B, N, T]
        outputs = [x]
        cur = x

        for layer_idx in range(self.n_down_layers):
            if self.down_sampling_method == "max":
                cur = F.max_pool1d(
                    cur,
                    kernel_size=self.down_sampling_window,
                    stride=self.down_sampling_window,
                )
            elif self.down_sampling_method == "avg":
                cur = F.avg_pool1d(
                    cur,
                    kernel_size=self.down_sampling_window,
                    stride=self.down_sampling_window,
                )
            elif self.down_sampling_method == "conv":
                cur = self.down_pools[layer_idx](cur)
            else:
                raise ValueError(
                    "down_sampling_method must be one of {'avg', 'max', 'conv'}, "
                    f"but got {self.down_sampling_method}."
                )

            cur = self._resize_time(cur, self.scale_lens[layer_idx + 1])
            outputs.append(cur)

        return outputs

    def _resize_spectrum(self, x, target_len):
        # x: [B, N, T_s]
        if x.size(-1) == target_len:
            return x
        return F.interpolate(x, size=target_len, mode="linear", align_corners=False)

    def TDT(self, x):

        B, T, N = x.shape
        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        x = self.enc_embedding(x, None)

        x = x.permute(0, 2, 1)
        x = self.embeddingMixer2(x)
        x = x.permute(0, 2, 1)
        x = self.embeddingMixer(x)
        x = x.permute(0, 2, 1)

        x = x.reshape(B, N, self.pred_len).contiguous()

        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        x_enc, means, stdev = self._normalize(x_enc)
        x = x_enc.permute(0, 2, 1).contiguous()  # [B, N, T]
        scale_inputs = self._multi_scale_downsample(x)

        scale_features = []
        for x_scale, block in zip(scale_inputs, self.scale_blocks):
            freq_scale = dct.dct(x_scale, norm="ortho")
            freq_scale = block(freq_scale)
            freq_scale = self._resize_spectrum(freq_scale, self.seq_len)
            scale_features.append(freq_scale)


        fused_freq = self.fusion(scale_features)
        future_freq = self.pred_head(fused_freq)

        dec_out = dct.idct(future_freq, norm="ortho").permute(0, 2, 1).contiguous()
        x_t = self.TDT(x_enc)

        Y = self.alpha * dec_out + (1-self.alpha) * x_t

        return self._denormalize(Y, means, stdev)

    def imputation(self, x_enc):
        return self.forecast(x_enc)

    def anomaly_detection(self, x_enc):
        return self.forecast(x_enc)

    def classification(self, x_enc):
        enc_out = self.forecast(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len :, :]
        if self.task_name == "imputation":
            return self.imputation(x_enc)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        if self.task_name == "classification":
            return self.classification(x_enc)
        return None
