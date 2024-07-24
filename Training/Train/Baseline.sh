# Note! 
# You will need to change the 2 data dirs below for each experiment attempt.

# python Baseline_FT.py \
#     --lr 0.00001 \
#     --batch_size 64 \
#     --num_epoch 300 \
#     --patience 4\
#     --num_cont 157 \
#     --num_cat 9 \
#     --dim 78 \
#     --dim_head 52 \
#     --depth 5 \
#     --heads 4 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint" \
#     --log_dir "./log" \
#     --result_dir "./result" \
#     --mode "train" \
#     --mimic_data_dir "C:\Users\DAHS\Desktop\ECP_CONT\ECP_SCL\Case Labeling\mimic_analysis.csv.gz" \
#     --eicu_data_dir "C:\Users\DAHS\Desktop\ECP_CONT\ECP_SCL\Case Labeling\eicu_analysis.csv.gz" 


python Baseline_MLP.py \
    --lr 0.00001 \
    --batch_size 64 \
    --num_epoch 300 \
    --patience 4\
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint" \
    --log_dir "./log" \
    --result_dir "./result" \
    --mode "Inference" \
    --mimic_data_dir "C:\Users\DAHS\Desktop\ECP_CONT\ECP_SCL\Case Labeling\mimic_analysis.csv.gz" \
    --eicu_data_dir "C:\Users\DAHS\Desktop\ECP_CONT\ECP_SCL\Case Labeling\eicu_analysis.csv.gz" 

    