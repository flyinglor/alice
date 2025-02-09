#!/bin/bash
#SBATCH -J ft_atp_ukb_dzne_predefine_points_ckp040_bs8_lr1e-4
#SBATCH -N 1
#SBATCH -p mcml-hgx-a100-80x4
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

srun python main-finetune.py \
    --dzne_dataset \
    --atp \
    --pretrain_ds "ukb" \
    --predefine_points \
    --roi_x 64 \
    --roi_y 64 \
    --roi_z 64 \
    --lr 0.0001 \
    --min_lr 1e-6 \
    --epochs 1000 \
    --warmup_epochs 20 \
    --saveckp_freq 1000 \
    --dist_url tcp://localhost:13155 \
    --data_dir "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/DZNE/" \
    --output_dir "/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/MSc/Hui/alice/dzne/atp_64_lr1e-4" \
    --checkpoint_file "/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/MSc/Hui/alice/ukb/roi64/checkpoint0040.pth"
