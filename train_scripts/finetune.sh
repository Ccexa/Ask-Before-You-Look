#!/bin/bash

echo "Finetune on Reason Seg"
echo "Start training"


CUDA_VISIBLE_DEVICES=2,3 \
deepspeed --master_port=25000 finetune_compass.py \
--version \path\to\pretrained_segmentation_COMPASS_Checkpoint \
--exp_name exp_name \
--batch_size 4 \
--grad_accumulation_steps 8 \
--epochs 2 \
--dataset_dir /path/to/data \
--dataset 'sem_seg||vqa||reason_seg' \
--sem_seg_data 'aggc2022_10x_1024||cctgs_10x_1024||conic' \
--reason_seg_data 'ReasonPath|train' \
--sample_rates 8,2,1 \
--vision_pretrained pretrain/sam_vit_h_4b8939.pth \
--val_dataset 'aggc2022_10x_1024|cctgs_10x_1024|conic' \
--lr 0.0001
