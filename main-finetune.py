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
from models.nnFormer import nnFormer, Encoder
from interfaces import init_model, get_embedding, find_point_in_vol
import pickle
import pprint
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter

from models.head import AliceHead, ClassificationHead2FC, ClassificationHead3FC, ClassificationHeadCLS
from loader import get_loader, get_loader_adni_ukb, get_loader_hos_dzne
from loss import Loss
from CASA import CASA_Module
from engine_pretrain import train_one_epoch
from validation import validation
#Dp
from torch.multiprocessing import Process
import torch.utils.data.distributed
import torch.distributed as dist
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from models.attentive_pooler import AttentiveClassifier
import tempfile

# from evaluation.unsupervised.unsup_cls import eval_pred
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

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
    parser.add_argument('--roi_x', default=32, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=32, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=32, type=int, help='roi size in z direction')
    parser.add_argument('--batch_size', default=8, type=int, help='number of batch size')
    parser.add_argument('--sw_batch_size', default=2, type=int, help='number of sliding window batch size')
    parser.add_argument('--normal_dataset', default=False, action='store_true', help='use monai Dataset class')
    parser.add_argument('--smartcache_dataset', default=False, action='store_true', help='use monai smartcache Dataset class')
    parser.add_argument('--dzne_dataset', default=False, action='store_true', help='use adni Dataset class')
    parser.add_argument('--hos_dataset', default=False, action='store_true', help='use ukb Dataset class')

    parser.add_argument('--distributed', action='store_true', default=True, help='enable distributed training')
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
    parser.add_argument('--atp', default=False, action='store_true', help='whether to use attentive pooling')
    parser.add_argument('--pretrain_ds', default="adni")
    parser.add_argument('--fc2', default=False, action='store_true', help='number of fc layers for finetuning')
    parser.add_argument('--fc3', default=False, action='store_true', help='number of fc layers for finetuning')
    parser.add_argument('--contrastive', default=False, action='store_true')
    parser.add_argument('--resize', default=False, action='store_true')
    parser.add_argument('--scratch', default=False, action='store_true')
    parser.add_argument('--predefine_points', default=False, action='store_true')
    parser.add_argument('--CLS', default=False, action='store_true')

    return parser

def train_FT(args, f=1):
   
    # utils.fix_random_seeds(args.seed)
    
    # ============ preparing data ... ============
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    # data_loader, test_loader = get_loader(args)

    #TODO load dzne, hospital dataset
    train_loader = get_loader_hos_dzne(args, fold=f, mode="train")
    val_loader =  get_loader_hos_dzne(args, fold=f, mode="val")
    test_loader = get_loader_hos_dzne(args, fold=f, mode="test")

    print(f"Data loaded: there are {len(train_loader)} train_loader.")
    print(f"Data loaded: there are {len(val_loader)} val_loader.")
    print(f"Data loaded: there are {len(test_loader)} test_loader.")

    # ============ building student and teacher networks ... ============
    # encoder = Encoder(
    #         img_size=(args.roi_x, args.roi_y, args.roi_z), 
    #         embed_dim=args.feature_size,
    #         depths=[2, 2, 2, 2],
    #         num_heads=[6, 12, 24, 48],
    #         patch_size=[2, 2, 2],
    #         window_size=[4, 4, 8, 4],
    #         in_chans=args.in_channels
    #         ).to(device)
    student = nnFormer(
              img_size=(args.roi_x, args.roi_y, args.roi_z),
              input_channels=args.in_channels,
              output_channels=args.out_channels,
              embedding_dim=args.feature_size,
              )
   
    embed_dim = args.feature_size * (2 ** 3)
    
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student, 
        AliceHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head,
        ),
    )
    # move networks to gpu
    student = student.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True) if \
        'nnformer' in args.arch else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
   
    # encoder = nn.parallel.DistributedDataParallel(encoder)
    # encoder = nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu], find_unused_parameters=True)
    # #TODO load student from checkpoint
    if not args.scratch:
        print("Loading the model from: ", args.checkpoint_file)
        checkpoint = torch.load(args.checkpoint_file, map_location="cuda" if torch.cuda.is_available() else "cpu")
        # Load state_dict
        student.load_state_dict(checkpoint['student'])

    # # Extract state dict for student
    # student_state_dict = checkpoint['student']
    # # Remap keys to match what the encoder expects
    # remapped_state_dict = {}
    # for k, v in student_state_dict.items():
    #     # Remap 'module.backbone.model_down' to 'module'
    #     if k.startswith("module.backbone.model_down."):
    #         new_key = k.replace("module.backbone.model_down.", "module.")
    #         remapped_state_dict[new_key] = v

    # # Load the remapped state dict into the encoder
    # encoder.load_state_dict(remapped_state_dict, strict=False)

    if args.atp:
        classifier = AttentiveClassifier(
            embed_dim=args.out_dim,
            # num_heads=encoder.module.num_heads,
            num_heads=8, 
            depth=1,
            num_classes=3
        )
    elif args.fc2: 
        #TODO classification head 
        #or wrap student and classification head together.
        # TODO add layernorm and pooling and try again
        classifier = ClassificationHead2FC(input_dim=args.out_dim, num_classes=3)
    elif args.CLS:
        classifier = ClassificationHeadCLS(input_dim=args.out_dim, num_classes=3)
    else:
        classifier = ClassificationHead3FC(input_dim=args.out_dim, num_classes=3)

    criterion = nn.CrossEntropyLoss()
    

    classifier = classifier.cuda()

    # TODO frozen backbone or not
    # Freeze all parameters by default
    for param in student.parameters():
        param.requires_grad = False


    # Unfreeze parameters in student.module.backbone.model_down except those starting with "backbone.model_down.norm"
    for name, param in student.module.backbone.model_down.named_parameters():
        if not name.startswith("norm"):  # Check if the name does not start with "norm"
            param.requires_grad = True


    # print(f"Encoder is built.")
    print(f"Student is built: it is {args.arch} network.")
        
    # ============ preparing optimizer ... ============
    # params_groups = utils.get_params_groups_dual(encoder, classifier)
    params_groups = utils.get_params_groups_dual(student, classifier)
    # params_groups = utils.get_params_groups(student)
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr, #* (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, 
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, 
        args.weight_decay_end,
        args.epochs, 
        len(train_loader),
    )
                  
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            student=student,
            classifier=classifier,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            alice_loss=alice_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting finetuning Alice!")
    
    best_val = 1e8
    
    iters = 0 # global training iteration
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        # print("epoch: ", epoch)

        # ============ training one epoch ... ============
        for itr, batch_data in enumerate(train_loader):
            # print("iteration: ", itr)
            imgs = batch_data['img'].to(device, non_blocking=True)
            # print("imgs.shape: ", imgs.shape)

            if not args.resize:
                #random crop
                if args.predefine_points:
                    pts = utils.select_predefined_points(1, imgs.transpose(2, 4))
                else:
                    pts = utils.select_random_points(1, imgs.transpose(2, 4))
                pts1 = pts[0]
                imgs =  utils.crop_tensor_new(imgs, pts1, args.roi_x, args.roi_y, args.roi_z).to(device)

            target = batch_data['label'].to(device, non_blocking=True)
            target = torch.argmax(target, dim=1)
            target = target.long()

            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[iters]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[iters]

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                
                # encoder_output = encoder(imgs)
                # en_feat = encoder_output[0][-1]

                student_output = student(imgs)
                clstoken, en_feat = student_output[0], student_output[1]
                # print("clstoken shape: ", clstoken.shape) # torch.Size([8, 512])
                # en_feat = student_output[1]
                #TODO try using both for classification task
                # print("en_feat shape: ", en_feat.shape) # torch.Size([8, 8, 512])
                if args.CLS:
                    outputs = classifier(clstoken)
                else:
                    outputs = classifier(en_feat)
                # outputs = classifier(clstoken)
                # print("Outputs shape:", outputs.shape)  # Should be (batch_size, num_classes)
                # print("Target shape:", target.shape)   # Should be (batch_size,)
                # print("Target dtype:", target.dtype)   # Should be torch.long
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                # if args.clip_grad:
                #     param_norms = utils.clip_gradients(student, args.clip_grad)
                # utils.cancel_gradients_last_layer(epoch, student,
                #                                 args.freeze_last_layer)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                # if args.clip_grad:
                #     fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                #     param_norms = utils.clip_gradients(student, args.clip_grad)
                # utils.cancel_gradients_last_layer(epoch, student,
                #                                 args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()


            if not args.disable_wandb:
                wandb.log(
                    {
                    f"Fold {f} - lr": optimizer.param_groups[0]['lr'],
                    f"Fold {f} - Training Loss": loss,
                    "custom_step": iters,
                    },
                )
            iters += 1

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            "classifier": classifier.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        
        if args.eval_epoch and (epoch % args.eval_epoch == 0):
            # iterate over validation dataloader
            val_losses = []
            for itr, batch_data in enumerate(val_loader):
                imgs = batch_data['img'].to(device, non_blocking=True)
                if not args.resize:
                    #random crop
                    if args.predefine_points:
                        pts = utils.select_predefined_points(1, imgs.transpose(2, 4))
                    else:
                        pts = utils.select_random_points(1, imgs.transpose(2, 4))
                    pts1 = pts[0]
                    imgs =  utils.crop_tensor_new(imgs, pts1, args.roi_x, args.roi_y, args.roi_z).to(device)

                target = batch_data['label'].to(device, non_blocking=True)
                target = torch.argmax(target, dim=1)
                target = target.long()
                with torch.cuda.amp.autocast(fp16_scaler is not None):
                    with torch.no_grad():
                        student_output = student(imgs)
                        clstoken, en_feat = student_output[0], student_output[1]
                        #TODO try using both for classification task
                        if args.CLS:
                            outputs = classifier(clstoken)
                        else:
                            outputs = classifier(en_feat)
                loss = criterion(outputs, target)    
                val_losses.append(float(loss))

                assert not np.isnan(float(loss)), 'loss is nan'

            # logging
            if not args.disable_wandb:
                wandb.log(
                    {
                    f"Fold {f} - Validation Loss": np.average(val_losses),
                    "custom_step": iters,
                    },
                )

            if np.average(val_losses) < best_val:
                best_val = np.average(val_losses)
                utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint_bestval.pth'))
                print('Model was saved ! Best Val Loss: {}'.format(best_val))
            else:
                print('Model was not saved ! Best Val Loss: {}'.format(best_val))
        
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        
    #TESTING
    test_losses = []
    predictions = []
    label_test = []
    #iterate over test dataloader
    for itr, batch_data in enumerate(test_loader):
        imgs = batch_data['img'].to(device, non_blocking=True)

        if not args.resize:
            #random crop
            if args.predefine_points:
                pts = utils.select_predefined_points(1, imgs.transpose(2, 4))
            else:
                pts = utils.select_random_points(1, imgs.transpose(2, 4))
            pts1 = pts[0]
            imgs =  utils.crop_tensor_new(imgs, pts1, args.roi_x, args.roi_y, args.roi_z).to(device)

        target = batch_data['label'].to(device, non_blocking=True)
        target = torch.argmax(target, dim=1)
        target = target.long()
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            with torch.no_grad():
                student_output = student(imgs)
                clstoken, en_feat = student_output[0], student_output[1]
                #TODO try using both for classification task
                if args.CLS:
                    outputs = classifier(clstoken)
                else:
                    outputs = classifier(en_feat)
            predictions.extend(torch.argmax(outputs, dim=1).tolist())
            label_test.extend(target.tolist())

        loss = criterion(outputs, target)    
        test_losses.append(float(loss))

    accuracy = accuracy_score(label_test, predictions)
    bal_acc = balanced_accuracy_score(label_test, predictions)
    precision = precision_score(label_test, predictions, average='macro')
    recall = recall_score(label_test, predictions, average='macro')
    f1 = f1_score(label_test, predictions, average='macro')


    log_string = f" ===> Test loss: {np.average(test_losses):.05f} \n "
    log_string += f"===> Accuracy: {accuracy:.05f} \n "
    log_string += f"===> Balanced Accuracy: {bal_acc:.05f} \n "
    log_string += f"===> Precision: {precision:.05f} \n "
    log_string += f"===> Recall: {recall:.05f} \n "
    log_string += f"===> F1-score: {f1:.05f} \n "
    log_string += f"===> Predictions: {predictions} \n "
    log_string += f"===> Labels: {label_test} \n "

    print(log_string)

    return np.average(test_losses), accuracy, bal_acc, precision, recall, f1
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Alice', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(args)

    if not args.disable_wandb:
        wandb_id = wandb.util.generate_id()
        ds = "dzne" if args.dzne_dataset else "hospital"
        if args.atp:
            ft = "atp"
        elif args.fc2:
            ft = "2fc"
        elif args.CLS:
            ft = "cls"
        else:
            ft = "3fc"

        pretrain_type = "contra" if args.contrastive else "all"
        pretrain_type = "scratch" if args.scratch else pretrain_type

        run = wandb.init(project=f"alice_ft_{ds}", 
                        name=f"{pretrain_type}_{args.pretrain_ds}_{ft}_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_{args.roi_x}^3", 
                        id=wandb_id,
                        resume='allow',
                        # dir=args.output_dir
                        dir=tempfile.mkdtemp() 
                        )



    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # 5 folds 
    fivefolds_test_loss = []
    fivefolds_test_accuracy = []
    fivefolds_test_bal_accuracy = []
    fivefold_test_precision = []
    fivefold_test_recall = []
    fivefold_test_f1 = []

    for f in range(1,6):
        if not args.disable_wandb:
            # define which metrics will be plotted against it
            wandb.define_metric(f"Fold {f} - lr", step_metric="custom_step")
            wandb.define_metric(f"Fold {f} - Training Loss", step_metric="custom_step")
            wandb.define_metric(f"Fold {f} - Validation Loss", step_metric="custom_step")

        #when do we use the folds? dataloader?
        print(f"################ Fold {f} ####################")
        test_loss, accuracy, bal_acc, precision, recall, f1= train_FT(args, f=f)

        fivefolds_test_loss.append(test_loss)
        fivefolds_test_accuracy.append(accuracy)
        fivefolds_test_bal_accuracy.append(bal_acc)
        fivefold_test_precision.append(precision)
        fivefold_test_recall.append(recall)
        fivefold_test_f1.append(f1)

    print(f"Average Test Loss: {np.mean(fivefolds_test_loss)}, std: {np.std(fivefolds_test_loss)}")
    print(f"Average Test Accuracy: {np.mean(fivefolds_test_accuracy)}, std: {np.std(fivefolds_test_accuracy)}")
    print(f"Average Test Balanced Accuracy: {np.mean(fivefolds_test_bal_accuracy)}, std: {np.std(fivefolds_test_bal_accuracy)}")
    print(f"Average Test Precision: {np.mean(fivefold_test_precision)}, std: {np.std(fivefold_test_precision)}")
    print(f"Average Test Recall: {np.mean(fivefold_test_recall)}, std: {np.std(fivefold_test_recall)}")
    print(f"Average Test F1: {np.mean(fivefold_test_f1)}, std: {np.std(fivefold_test_f1)}")

    if args.local_rank == 0 and not args.disable_wandb:
        run.finish()
