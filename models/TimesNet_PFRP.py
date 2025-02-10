import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.TimesNet import TimesNet
from utils.tools import plot_retrieve


def normalize_series(tensor):
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    std[std == 0] = 1e-6
    normalized_sequences = (tensor - mean) / std
    return mean, std, normalized_sequences


def denormalize_series(mean, std, normalized_sequences):
    original_sequences = normalized_sequences * std + mean
    return original_sequences


class PastSeriesEncoder(nn.Module):
    def __init__(self, input_dim=96, output_dim=16):
        super(PastSeriesEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)


class CustomMLP(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(CustomMLP, self).__init__()
        hidden_dim = 2 * seq_len
        self.fc1 = nn.Linear(seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*pred_len)
        
        self._init_weights(pred_len)

    def _init_weights(self, pred_len):
        with torch.no_grad():
            self.fc2.weight.data.fill_(0) 
            self.fc2.bias.data[:pred_len].fill_(1)
            self.fc2.bias.data[pred_len:].fill_(0)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class TimesNet_PFRP(nn.Module):
    def __init__(self, configs):
        super(TimesNet_PFRP, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.topk = configs.topk_PFRP
        dataset_name = configs.data_path.split('.')[0]

        self.series_encoder = PastSeriesEncoder(self.seq_len)
        self.series_encoder.load_state_dict(torch.load('./checkpoints_CL/checkpoint_' + dataset_name + '.pth'))
        for param in self.series_encoder.parameters():
            param.requires_grad = False

        self.confidence_gate = nn.Sequential(
            nn.Linear(self.seq_len+self.pred_len, configs.d_ff_PFRP),
            nn.GELU(),
            nn.Linear(configs.d_ff_PFRP, 1),
            nn.Sigmoid()
        )

        self.output_gate = CustomMLP(self.seq_len, self.pred_len)

        self.predictor = TimesNet(configs)

        self.weight_mlp = nn.Sequential(
            nn.Linear(self.topk, configs.d_ff_PFRP),
            nn.GELU(),
            nn.Linear(configs.d_ff_PFRP, 2),
            nn.Softmax(dim=-1)
        )
        
        # Load GMB
        past_features = torch.tensor(np.load('./GMB/'+dataset_name+'/feature_96.npy'))
        self.register_buffer('past_features', past_features)
        past_series = torch.tensor(np.load('./GMB/'+dataset_name+'/past_96.npy'))
        self.register_buffer('past_series', past_series)
        future_series = torch.tensor(np.load('./GMB/'+dataset_name+'/future_96_720.npy'))
        self.register_buffer('future_series', future_series[:, :self.pred_len])

    def forecast(self, x, x_mark_enc, x_dec, x_mark_dec):
        y1 = self.predictor(x, x_mark_enc, x_dec, x_mark_dec)
        mean, std, normalized_sequences = normalize_series(x.squeeze(-1))

        features = self.series_encoder(normalized_sequences)          # (B, D)
        features = F.normalize(features, dim=1)

        cosine_similarity = torch.matmul(features, self.past_features.T) # (B, N)
        topk_similarity, topk_indices = torch.topk(cosine_similarity, k=self.topk, dim=1)  # (B, K), (B, K)
        topk_similarity_raw = topk_similarity

        topk_past = self.past_series[topk_indices]
        topk_future = self.future_series[topk_indices]        # (B, K, L)

        # with confidence gate
        conf_gate = self.confidence_gate(torch.cat([normalized_sequences.unsqueeze(1).expand(-1, self.topk, -1), topk_future], dim=-1))  # (B, K)
        topk_similarity = topk_similarity * conf_gate.squeeze(-1)    # (B, K)

        topk_similarity_softmax = torch.softmax(topk_similarity, dim=1)
        topk_similarity_softmax = topk_similarity_softmax.unsqueeze(-1)
        fusion_future = (topk_future * topk_similarity_softmax).sum(dim=1)  # (B, L)

        # plot_retrieve(normalized_sequences, topk_past, topk_future)

        # with output gate
        out_gate = self.output_gate(normalized_sequences)
        out_scale = out_gate[:, :self.pred_len]
        out_shift = out_gate[:, self.pred_len:]
        seasonal_output = fusion_future * out_scale + out_shift #

        y = denormalize_series(mean, std, seasonal_output).unsqueeze(-1)

        weight = self.weight_mlp(topk_similarity).unsqueeze(1)
        pred = torch.sum(torch.cat([y, y1], dim=-1) * weight, dim=-1)

        return pred.unsqueeze(-1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]