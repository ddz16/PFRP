import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import numpy as np
import time  
from data_provider.data_factory import data_provider


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

def normalize_series(tensor):
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    std[std == 0] = 1e-6
    normalized_sequences = (tensor - mean) / std
    return normalized_sequences

def contrastive_loss(features, x, x_mark, y, y_mark, temperature=0.05, min_time_gap=4):
    # pos or neg: mse between y
    # exclude close timestamp 
    B, D = features.size()
    L = x.size(1)

    mse_matrix = torch.cdist(y, y, p=2) # (B, B)

    # exclude close timestamp and itself
    start_times = x_mark[:, 0]
    time_diffs = torch.abs(start_times.unsqueeze(1) - start_times.unsqueeze(0))  # (B, B)
    mask = time_diffs < min_time_gap * 60 * 60  # (B, B)
    mse_matrix[mask] = 1e6

    # positive samples
    positive_indices = torch.argmin(mse_matrix, dim=1)  # (B,)

    # compute feature similarity
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / temperature  # (B, B)
    logits_with_mask = torch.where(mask, torch.tensor(float('-inf')).to(logits.device), logits)
    # logits_with_mask = logits - mask.float() * 1e6

    # contrastive loss
    loss = F.cross_entropy(logits_with_mask, positive_indices)

    return loss


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Contrastive Learning')

    parser.add_argument('--task_name', type=str, default='long_term_forecast')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    # parser.add_argument('--root_path', type=str, default='./dataset/traffic/', help='root path of the data file')
    # parser.add_argument('--data_path', type=str, default='traffic.csv', help='data file')
    parser.add_argument('--root_path', type=str, default='./dataset/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_CL/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
   
    # optimization
    parser.add_argument('--num_workers', type=int, default=16, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=80, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--temperature', type=float, default=0.05, help="temperature in contrastive loss")
    parser.add_argument('--min_time_gap', type=int, default=4, help="exclude close timestamp")

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_data, train_loader = data_provider(args, flag='train', timeenc=2)

    pse = PastSeriesEncoder(args.seq_len).to(device)
    optimizer = torch.optim.Adam(pse.parameters(), lr=args.learning_rate)

    start_time = time.time()

    for epoch in range(args.train_epochs):
        pse.train()
        total_loss = 0

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
            batch_x = batch_x.float().squeeze(-1).to(device)                 # (B, L)
            batch_x_mark = batch_x_mark.float().squeeze(-1).to(device)    # (B, L)
            batch_y = batch_y.float().squeeze(-1).to(device)
            batch_y = batch_y[:, -args.pred_len:]

            batch_x = normalize_series(batch_x)
            batch_y = normalize_series(batch_y)
            x_features = pse(batch_x)  # (B, D)
            loss = contrastive_loss(x_features, batch_x, batch_x_mark, batch_y, batch_y_mark, args.temperature, args.min_time_gap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    
    torch.save(pse.state_dict(), args.checkpoints + 'checkpoint_' + args.data_path.split('.')[0] + '_seq720.pth')
    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Consumed time: {elapsed_time:.2f} seconds!")

