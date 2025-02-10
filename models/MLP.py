import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_series(tensor):
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    std[std == 0] = 1e-6
    normalized_sequences = (tensor - mean) / std
    return mean, std, normalized_sequences

def denormalize_series(mean, std, normalized_sequences):
    original_sequences = normalized_sequences * std + mean
    return original_sequences


class MLP(nn.Module):
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.model = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.head = nn.Linear(16, self.pred_len)

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        mean, std, normalized_sequences = normalize_series(x_enc.squeeze(-1))
        dec_out = self.head(self.model(normalized_sequences))
        y = denormalize_series(mean, std, dec_out).unsqueeze(-1)
        return y[:, -self.pred_len:, :]  # [B, L, D]