export CUDA_VISIBLE_DEVICES=0

model_name=SparseTSF_PFRP

root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

seq_len=96
for pred_len in 96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features S \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 1 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 256 --learning_rate 0.03 --d_ff_PFRP 128 --topk_PFRP 50
done
