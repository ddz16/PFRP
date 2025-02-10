export CUDA_VISIBLE_DEVICES=1

model_name=SparseTSF_PFRP

root_path_name=./dataset/ETT-small/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

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
    --period_len 4 \
    --enc_in 1 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --d_ff_PFRP 128 --topk_PFRP 200
done
