export CUDA_VISIBLE_DEVICES=0
cd src/ai_pcc/GausPcgc

python train.py \
    --training_data='/data/trainset_quan/*.ply' \
    --val_data='/data/quantized/*.ply' \
    --model_save_folder='./model/ue_4stage_conv_k5_Q1_patch' \
    --is_data_pre_quantized=True \
    --channels=32 \
    --kernel_size=5 \
    --batch_size=1 \
    --stage='ue_4stage_conv' \
    --log_folder "./logs/ue_4stage_conv_k5"