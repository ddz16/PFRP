import os
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_provider.data_factory import data_provider
from sklearn_extra.cluster import KMedoids


parser = argparse.ArgumentParser(description='Construct GMB')
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
parser.add_argument('--colors', type=int, default=24)

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

# Augmentation
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--K', type=int, default=2000, help='K Medoids')
args = parser.parse_args()
dataset_name = args.data_path.split('.')[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def normalize_numpy(sequences):
    mean = sequences.mean(axis=1, keepdims=True)  # mean (B*1)
    std = sequences.std(axis=1, keepdims=True)    # std (B*1)
    std[std == 0] = 1e-6

    normalized_sequences = (sequences - mean) / std
    return normalized_sequences


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


start_time = time.time()

train_data, train_loader = data_provider(args, flag='train', timeenc=2, construct_GMB=True)
batch_x_list = []
batch_y_list = []
for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
    batch_x = batch_x.float().squeeze(-1)               # (B, L)
    batch_y = batch_y.float().squeeze(-1)
    batch_y = batch_y[:, -args.pred_len:]
    batch_x_list.append(batch_x)
    batch_y_list.append(batch_y)

sequences = torch.cat(batch_x_list, dim=0).numpy()
sequences = normalize_numpy(sequences) 
future_sequences = torch.cat(batch_y_list, dim=0).numpy()
future_sequences = normalize_numpy(future_sequences) 

#####  Load Encoder  #####
model = PastSeriesEncoder(args.seq_len).to(device)
model.load_state_dict(torch.load(os.path.join(args.checkpoints, "checkpoint_"+dataset_name+".pth")))  
# saved_weights = torch.load(os.path.join(args.checkpoints, "checkpoint_"+dataset_name+"_supervised.pth"))
# model_weights = {k.replace('model.', ''): v for k, v in saved_weights.items() if k.startswith('model.')}
# model.model.load_state_dict(model_weights)
model.eval()

with torch.no_grad():
    features = model(torch.tensor(sequences).to(device))
    features = F.normalize(features, dim=1).detach().cpu().numpy()

tsne = TSNE(n_components=2, random_state=42, perplexity=30, metric='cosine')
tsne_results = tsne.fit_transform(features)

num_points = tsne_results.shape[0]
colors_per_group = args.colors
cmap = plt.cm.get_cmap('tab20', colors_per_group)
colors = np.array([i % colors_per_group for i in range(num_points)])


plt.figure(figsize=(10, 6))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, cmap=cmap, s=20)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
# plt.savefig('./GMB/'+dataset_name+'/all_features.pdf')

# get the indices of cluster centers
kmedoids = KMedoids(n_clusters=args.K, metric='cosine', random_state=5)
kmedoids.fit(features)
medoid_indices = kmedoids.medoid_indices_

plt.figure(figsize=(10, 6))
scatter = plt.scatter(tsne_results[medoid_indices, 0], tsne_results[medoid_indices, 1], c=colors[medoid_indices], cmap=cmap, s=20) #, cmap=cmap
plt.xticks([])
plt.yticks([])
plt.tight_layout()
# plt.savefig('./GMB/'+dataset_name+'/GMB_features.pdf')

arr = colors[medoid_indices]
values, counts = np.unique(arr, return_counts=True)
for value, count in zip(values, counts):
    print(f"The {value} appears {count} times!")

np.save('./GMB/'+dataset_name+f'/feature_{args.seq_len}.npy', features[medoid_indices, :])
np.save('./GMB/'+dataset_name+f'/past_{args.seq_len}.npy', sequences[medoid_indices, :])
np.save('./GMB/'+dataset_name+f'/future_{args.seq_len}_720.npy', future_sequences[medoid_indices, :])

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Consumed time: {elapsed_time:.2f} seconds!")
