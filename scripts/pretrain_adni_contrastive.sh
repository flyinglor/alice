#!/bin/bash
#SBATCH -J contrastive_vitb_adni_bs16_lr1e-3_64x64x64
#SBATCH -N 1
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --ntasks=1
#SBATCH --mail-user=hui.zheng@tum.de
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH -o %x.%j.%N.out

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate alice # activate your environment

cd ~/alice/

export WANDB_API_KEY=9b379393a7a65969e05ab4e01683be3b8770aabf

srun python main-DDP.py \
    --adni_dataset \
    --lambda3 0 \
    --arch vit_base \
    --batch_size 16 \
    --lr 0.001 \
    --min_lr 1e-5 \
    --epochs 1000 \
    --saveckp_freq 50 \
    --warmup_epochs 20 \
    --roi_x 64 \
    --roi_y 64 \
    --roi_z 64 \
    --dist_url tcp://localhost:13161 \
    --output_dir /dss/dssmcmlfs01/pr62la/pr62la-dss-0002/MSc/Hui/alice/adni/contrastive/vitb_roi64