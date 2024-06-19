# Note! 
# You will need to change the 2 data dirs below for each experiment attempt.

# Training - Optuna
python main.py \
    --lr 0.00001 \
    --batch_size 64 \
    --num_epoch 300 \
    --patience 4\
    --num_cont 157 \
    --num_cat 9 \
    --dim 78 \
    --dim_head 52 \
    --depth 5 \
    --heads 4 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint" \
    --log_dir "./log" \
    --result_dir "./result" \
    --mode "train" \
    --mimic_data_dir "/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz" \
    --eicu_data_dir "/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/eicu_analysis.csv.gz" 

# Get contrastive embedding
python main.py \
    --lr 0.00001 \
    --batch_size 64 \
    --num_epoch 300 \
    --patience 4\
    --num_cont 157 \
    --num_cat 9 \
    --dim 78 \
    --dim_head 52 \
    --depth 5 \
    --heads 4 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint" \
    --log_dir "./log" \
    --result_dir "./result" \
    --mode "Get_Embedding" \
    --mimic_data_dir "/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz" \
    --eicu_data_dir "/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/eicu_analysis.csv.gz" 


# Get Feature Importance
python main.py \
    --lr 0.00001 \
    --batch_size 64 \
    --num_epoch 300 \
    --patience 4\
    --num_cont 157 \
    --num_cat 9 \
    --dim 78 \
    --dim_head 52 \
    --depth 5 \
    --heads 4 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint" \
    --log_dir "./log" \
    --result_dir "./result" \
    --mode "Get_Feature_Importance" \
    --mimic_data_dir "/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz" \
    --eicu_data_dir "/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/eicu_analysis.csv.gz" 

    