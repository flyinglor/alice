#!/bin/bash
#SBATCH -J te_ft_3fc_ukb_hos_resize64^3_ckp040_bs8_lr1e-4
#SBATCH -N 1
#SBATCH -p mcml-dgx-a100-40x8 
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=128gb
#SBATCH --ntasks=1
#SBATCH --mail-user=hui.zheng@tum.de
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH -o %x.%j.%N.out

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate alice # activate your environment

cd ~/alice/

export WANDB_API_KEY=9b379393a7a65969e05ab4e01683be3b8770aabf

srun python te-finetune.py \
    --hos_dataset \
    --fc3 \
    --resize \
    --pretrain_ds "ukb" \
    --roi_x 64 \
    --roi_y 64 \
    --roi_z 64 \
    --lr 0.0001 \
    --min_lr 1e-6 \
    --weight_decay 1e-5 \
    --epochs 1000 \
    --warmup_epochs 50 \
    --saveckp_freq 1000 \
    --dist_url tcp://localhost:13156 \
    --data_dir "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/238+19+72_tum_splits/" \
    --output_dir "/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/MSc/Hui/alice/hos/atp_64_lr1e-4" \
    --checkpoint_file "/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/MSc/Hui/alice/ukb/roi64/checkpoint0040.pth"
