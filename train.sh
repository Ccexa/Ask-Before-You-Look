#!/bin/bash

echo "TEST FOR REASON SEG"
echo "Start training"




# finetune compass
CUDA_VISIBLE_DEVICES=0,1 \
deepspeed --master_port=25000 finetune_compass.py \
--version experiments/compass7b/weight \
--exp_name finetune_compass_10_31_test \
--batch_size 4 \
--grad_accumulation_steps 8 \
--epochs 2 \
--dataset_dir /data3/hxmao/dataset/lisa_data_v1 \
--dataset 'sem_seg||vqa||reason_seg' \
--sem_seg_data 'aggc2022_10x_1024||cctgs_10x_1024||conic' \
--reason_seg_data 'PathInstruct|train' \
--sample_rates 7,3,4 \
--vision_pretrained pretrain/sam_vit_h_4b8939.pth \
--val_dataset 'aggc2022_10x_1024|cctgs_10x_1024|conic' \
--lr 0.0001


# model training
# CUDA_VISIBLE_DEVICES=0,1 \
# deepspeed --master_port=25000 train_compass.py \
# --version ./pretrain/LLaVA-Lightning-7B-v1-1 \
# --exp_name compass_7b_1026 \
# --batch_size 4 \
# --grad_accumulation_steps 8 \
# --epochs 12 \
# --dataset_dir /data3/hxmao/dataset/lisa_data_v1 \
# --dataset 'sem_seg||vqa' \
# --sem_seg_data 'aggc2022_10x_1024||cctgs_10x_1024||conic' \
# --sample_rates 7,3 \
# --vision_pretrained pretrain/sam_vit_h_4b8939.pth \
# --val_dataset 'aggc2022_10x_1024|cctgs_10x_1024|conic' \
# --lr 0.0001

# TEST ON REASON SEG
# CUDA_VISIBLE_DEVICES=1 \
# deepspeed --master_port=25000 train_compass.py \
# --version ./pretrain/LLaVA-Lightning-7B-v1-1 \
# --exp_name test_reason_seg_10_30 \
# --batch_size 1 \
# --grad_accumulation_steps 8 \
# --epochs 2 \
# --dataset_dir /data3/hxmao/dataset/lisa_data_v1 \
# --dataset 'sem_seg||vqa||reason_seg' \
# --sem_seg_data 'aggc2022_10x_1024||cctgs_10x_1024||conic' \
# --reason_seg_data 'PathInstruct|train' \
# --sample_rates 7,3,4 \
# --vision_pretrained pretrain/sam_vit_h_4b8939.pth \
# --val_dataset 'aggc2022_10x_1024|cctgs_10x_1024|conic' \
# --lr 0.0001

# echo "ABLATION STUDY ON WO TUMOR VQA ON LISA-7B-SAM-H"
# CUDA_VISIBLE_DEVICES=0,1 \
# deepspeed --master_port=25000 train_path_seg.py \
# --version ./pretrain/LLaVA-Lightning-7B-v1-1 \
# --exp_name LISA-7B_SAM_H_1029_wo_tumor_vqa \
# --batch_size 32 \
# --grad_accumulation_steps 1 \
# --epochs 12 \
# --dataset_dir /data3/hxmao/dataset/lisa_data_v1 \
# --dataset 'sem_seg||vqa' \
# --sem_seg_data 'aggc2022_10x_1024||cctgs_10x_1024' \
# --sample_rates 7,3 \
# --vision_pretrained pretrain/sam_vit_h_4b8939.pth \
# --val_dataset 'aggc2022_10x_1024|cctgs_10x_1024' \
# --lr 0.0002

# LISA-7B-SAM-H
# CUDA_VISIBLE_DEVICES=0,1 \
# deepspeed --master_port=25000 train_path_seg.py \
# --version ./pretrain/LLaVA-Lightning-7B-v1-1 \
# --exp_name LISA-7B_SAM_H_1027 \
# --batch_size 32 \
# --grad_accumulation_steps 1 \
# --epochs 12 \
# --dataset_dir /data3/hxmao/dataset/lisa_data_v1 \
# --dataset 'sem_seg||vqa' \
# --sem_seg_data 'aggc2022_10x_1024||cctgs_10x_1024||conic' \
# --sample_rates 7,3 \
# --vision_pretrained pretrain/sam_vit_h_4b8939.pth \
# --val_dataset 'aggc2022_10x_1024|cctgs_10x_1024|conic' \
# --lr 0.0004

# COMPASS-7B
# CUDA_VISIBLE_DEVICES=0,1 \
# deepspeed --master_port=25000 train_compass.py \
# --version ./pretrain/LLaVA-Lightning-7B-v1-1 \
# --exp_name compass_7b_1026 \
# --batch_size 4 \
# --grad_accumulation_steps 8 \
# --epochs 12 \
# --dataset_dir /data3/hxmao/dataset/lisa_data_v1 \
# --dataset 'sem_seg||vqa' \
# --sem_seg_data 'aggc2022_10x_1024||cctgs_10x_1024||conic' \
# --sample_rates 7,3 \
# --vision_pretrained pretrain/sam_vit_h_4b8939.pth \
# --val_dataset 'aggc2022_10x_1024|cctgs_10x_1024|conic' \
# --lr 0.0001

#  ABL-COMPASS-BASE
# CUDA_VISIBLE_DEVICES=0,1 \
# deepspeed --master_port=25000 train_compass_base.py \
# --version ./pretrain/LLaVA-Lightning-7B-v1-1 \
# --exp_name compass_ablation_base_1027 \
# --batch_size 4 \
# --grad_accumulation_steps 8 \
# --epochs 12 \
# --dataset_dir /data3/hxmao/dataset/lisa_data_v1 \
# --dataset 'sem_seg||vqa' \
# --sem_seg_data 'aggc2022_10x_1024||cctgs_10x_1024||conic' \
# --sample_rates 7,3 \
# --vision_pretrained pretrain/sam_vit_h_4b8939.pth \
# --val_dataset 'aggc2022_10x_1024|cctgs_10x_1024|conic' \
# --lr 0.0001



# CUDA_VISIBLE_DEVICES=1 \
# deepspeed --master_port=24999 train_path_seg.py \
# --version ./pretrain/LLaVA-Lightning-7B-v1-1 \
# --exp_name test \
# --batch_size 16 \
# --grad_accumulation_steps 2 \
# --epochs 12 \
# --dataset_dir /data2/hxmao/dataset/dataset/lisa_data_v1 \
# --dataset 'sem_seg||vqa' \
# --sem_seg_data 'aggc2022_10x_1024||cctgs_10x_1024||conic' \
# --sample_rates 7,3 \
# --vision_pretrained pretrain/sam_vit_h_4b8939.pth \
# --val_dataset 'aggc2022_10x_1024|cctgs_10x_1024|conic' \
# --lr 0.0004 \
# --resume 'runs/lisa7b_aggc_cctgs_conic_1007_sam-h/ckpt_model' 



# finetuning
# CUDA_VISIBLE_DEVICES=2,3 \
# deepspeed --master_port=24999 train_path_seg.py \
# --version LISA-7B_base_llm_llama_0924_saml/LISA-7B_0924 \
# --exp_name lisa7b_conic_ft_based_0924_saml_1007 \
# --batch_size 32 \
# --grad_accumulation_steps 1 \
# --epochs 5 \
# --dataset_dir /data1/hxmao/dataset/lisa_data_v1 \
# --dataset 'sem_seg||vqa' \
# --sem_seg_data 'conic||aggc2022_10x_1024||cctgs_10x_1024' \
# --sample_rates 9,1 \
# --vision_pretrained LISA-7B_base_llm_llama_0924_saml/LISA-7B_0924/Lisa_tuned_SAM.bin \
# --val_dataset 'conic|aggc2022_10x_1024|cctgs_10x_1024' \
# --lr 0.0002 \




# CUDA_VISIBLE_DEVICES=1 \
# deepspeed --master_port=24999 train_path_seg.py \
# --version pretrain/LLaVA-Lightning-7B-v1-1 \
# --exp_name lisa7b_pretrain_semseg_vqa_val_debug_v3 \
# --batch_size 32 \
# --grad_accumulation_steps 1 \
# --epochs 1 \
# --dataset_dir /data/hxmao/dataset/lisa_data \
# --dataset 'sem_seg||vqa' \
# --sem_seg_data 'cctgs_thumbnails||aggc_thumbnails' \
# --sample_rates 9,1 \
# --vision_pretrained ./pretrain/sam_vit_h_4b8939.pth \
# --val_dataset 'cctgs_thumbnails|aggc_thumbnails' \
# --lr 0.0004

# inference
# CUDA_VISIBLE_DEVICES=1 \
# deepspeed --master_port=24999 train_path_seg.py \
#     --version ./LISA-7B_0924 \
#     --exp_name lisa7b_0924_aggc_cctgs_patch_eval \
#     --dataset_dir /data1/hxmao/dataset/lisa_data_v1 \
#     --dataset 'sem_seg||vqa' \
#     --sem_seg_data 'aggc2022_10x_1024||cctgs_10x_1024' \
#     --sample_rates 9,1 \
#     --vision_pretrained './pretrain/sam_vit_l_0b3195.pth' \
#     --val_dataset 'aggc2022_10x_1024|cctgs_10x_1024' \
#     --eval_only
