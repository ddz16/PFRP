# PFRP
This repository contains the pytorch code for the paper "Predicting the Future by Retrieving the Past”.

## Environment
Run the following command:
```
pip install -r requirements.txt
```

## Datasets
We use the datasets provided by [TSLib](https://github.com/thuml/Time-Series-Library). All the datasets are well pre-processed and can be used easily. After downloading, put these dataset files in the `./dataset/` folder.

## The First Stage
The first stage involves constructing a Global Memory Bank (GMB) to store historical information. 
If you want to construct a GMB for a dataset, you should train a lookback window encoder with Predictive Contrastive Learning (PCL) firstly. For example, you can run this command to train a encoder for the traffic dataset:
```
python run_CL.py --data custom --root_path ./dataset/traffic/ --data_path traffic.csv
```
The trained lookback window encoder will be saved in the `./checkpoints_CL/` folder. We have provided the checkpoints of the trained lookback window encoders for all the datasets in the `./checkpoints_CL/` folder. So you can directly use them.

After obtaining the trained encoder, you can extract lookback window features of all the training samples with it and construct the GMB:
```
python construct_GMB.py --root_path ./dataset/traffic/ --data_path traffic.csv --K 4000
```
Each GMB can be saved as three numpy arrays. `past_96.npy` represents K lookback window sequences, `feature_96.npy` represents the features of these lookback window sequences (i.e., the keys in GMB), and `future_96_720.npy` represents the corresponding prediction horizon sequence (i.e., the values in GMB). We have provided the GMBs for all the datasets in the `./GMB/` folder.

## The Second Stage
The second stage focuses on prediction through GMB retrieval, i.e., predicting the future by retrieving the past. You can train and evaluate all the prediction models with or without PFRP. We provide the experiment scripts for all models under the folder `./scripts/`. You can reproduce the experiment results as the following examples:
```
bash ./scripts/long_term_forecast/Traffic_script/SparseTSF.sh
bash ./scripts/long_term_forecast/Traffic_script/SparseTSF_PFRP.sh
```

## Citation
```
@inproceedings{pfrp,
  title={Predicting the Future by Retrieving the Past},
  author={Du, Dazhao and Han, Tao and Guo, Song},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

