import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm
from models.nnFormer import nnFormer
from interfaces import init_model, get_embedding, get_batch_embedding, find_point_in_vol
from sam.datasets.pipelines import Collect3d
from sam.datasets.collect import collate
import pickle

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter

from models.head import AliceHead
from loader import get_loader
from loss import Loss
from CASA import CASA_Module
from loader import get_loader, get_loader_adni_ukb

#Dp
from torch.multiprocessing import Process
import torch.utils.data.distributed
import torch.distributed as dist


def get_args_parser():
    parser = argparse.ArgumentParser('Alice', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='nnformer', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
                 'swin_tiny','swin_small', 'swin_base', 'swin_large'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=512, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=512, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--feature_size', default=48, type=int, help='feature size')
    parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
    parser.add_argument('--out_channels', default=1, type=int, help='number of output channels')
    
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for contrastive
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for contrastive 
        loss over patch token embeddings (Default: 1.0)""")
    parser.add_argument('--lambda3', default=10, type=float, help="""loss weight for MAE 
        loss over masked patch tokens (Default: 1.0)""")       
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.008, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.01, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.008, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.01, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--warmup_epochs", default=50, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument("--lr", default=5e-2, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)

    
    # Medical data process
    parser.add_argument('--a_min', default=-125, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=225, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--roi_x', default=64, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=64, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=64, type=int, help='roi size in z direction')
    parser.add_argument('--batch_size', default=16, type=int, help='number of batch size')
    parser.add_argument('--sw_batch_size', default=2, type=int, help='number of sliding window batch size')
    parser.add_argument('--normal_dataset', default=False, action='store_true', help='use monai Dataset class')
    parser.add_argument('--smartcache_dataset', default=False, action='store_true', help='use monai smartcache Dataset class')
    parser.add_argument('--adni_dataset', default=True, action='store_true', help='use adni Dataset class')
    parser.add_argument('--ukb_dataset', default=False, action='store_true', help='use ukb Dataset class')

    parser.add_argument('--distributed', action='store_true', default=False, help='enable distributed training')
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--mask_ratio', default=0.75, help='mask ratio')
    
    # Misc
    parser.add_argument("--data_dir", default="/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/ADNI", type=str, help="dataset directory")
    parser.add_argument("--json_list", default="/mnt/workspace/jiangyankai/Alice_code/datasets/pretrainset_test.json", type=str, help="dataset json file")
    parser.add_argument('--output_dir', default="./results/adni/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--embed_dir', default="/mnt/data/oss_beijing/jiangyankai/AbdomenAtlas_Prcessed/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--config_file', default="./configs/sam/sam_r18_i3d_fpn_1x_multisets_sgd_T_0.5_half_test.py", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--checkpoint_file', default="./checkpoints_SAM/SAM.pth", type=str, help='Path to save logs and checkpoints.')
    
    parser.add_argument('--saveckp_freq', default=100, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--eval_epoch', default=10, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="tcp://localhost:13147", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--master_port", default='29501', type=str, help="Please ignore and do not set this argument.")
    parser.add_argument("--gpu", default=0, type=int, help="Please ignore and do not set this argument.")
    
    parser.add_argument('--disable_wandb', default=False, action='store_true', help='whether to use wandb logging')
    
    return parser

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Alice', parents=[get_args_parser()])
    args = parser.parse_args()
    #dataloader or single image?
    data_loader, test_loader = get_loader_adni_ukb(args, test=True)

    # ============ building SAM model ... ============    
    sam_model, sam_cfg = init_model(args.config_file, args.checkpoint_file)

    for it, batch in enumerate(data_loader):
        image = batch['img']
        memory_queue_patch = batch
        
        memory_image = memory_queue_patch['img']

        norm_info_1 = np.array(sam_cfg.norm_spacing) / np.array(sam_cfg.norm_spacing)
        norm_info_2 = norm_info_1

        #get the embeddings from pretrained sam model
        emb1 = get_batch_embedding(sam_model, image[0], sam_cfg)
        emb2 = get_batch_embedding(sam_model, memory_image[1], sam_cfg)

        iter_points, scores = 0, 0
        while iter_points<=100 and scores<=0.7:
            pts = utils.select_random_points(2, image.transpose(2, 4))
            pts1, pts2 = pts[0], pts[1]
            pts1_pred, scores = find_point_in_vol(emb1, emb2, [pts1], sam_cfg)
            iter_points += 1
        # pts1_pred = pts1_pred[0]

        # # to save
        utils.visualize(image[0].squeeze(0).numpy(), memory_image[0].squeeze(0).numpy(), norm_info_1, norm_info_2, pts1, pts1_pred, scores, "./results/images/")
        # not save
        # utils.visualize(image[0].squeeze(0).numpy(), memory_image[0].squeeze(0).numpy(), norm_info_1, norm_info_2, pts1, pts1_pred, scores)

        pts1_pred = pts1_pred[0]
        query =  utils.crop_tensor_new(image, pts1, args.roi_x, args.roi_y, args.roi_z)
        # print(query.shape) # torch.Size([16, 1, 64, 64, 64])
        anchor = utils.crop_tensor_new(memory_image, pts1_pred, args.roi_x, args.roi_y, args.roi_z)
        # print(anchor.shape) # torch.Size([16, 1, 64, 64, 64])

        utils.visualize_crop(query[0].squeeze(0), anchor[1].squeeze(0), "./results/images/")

        # query_aug, anchor_aug= utils.data_aug(args, query), utils.data_aug(args, anchor)
        # images_normal = [query, anchor]
        # images_aug = [query_aug, anchor_aug]
        # masks = utils.random_mask(args, images_normal)