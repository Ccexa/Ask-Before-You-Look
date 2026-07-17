#!bin/bash

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX

# model training
CUDA_VISIBLE_DEVICES=0,1 \
deepspeed --master_port=25000 train_compass.py \
    --version ./pretrain/LLaVA-Lightning-7B-v1-1 \
    --exp_name your_exp_name \
    --batch_size 4 \
    --grad_accumulation_steps 8 \
    --epochs 12 \
    --dataset_dir /path/to/lisa_data_v1 \
    --dataset 'sem_seg||vqa' \
    --sem_seg_data 'aggc2022_10x_1024||cctgs_10x_1024||conic' \
    --sample_rates 7,3 \
    --vision_pretrained pretrain/sam_vit_h_4b8939.pth \
    --val_dataset 'aggc2022_10x_1024|cctgs_10x_1024|conic' \
    --lr 0.0001 
    # --vis \ 
