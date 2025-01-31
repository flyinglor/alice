import random
import math
import numpy as np
import os
import torch
import pandas as pd

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    RandSpatialCropSamplesd,
    SpatialPadd,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    ThresholdIntensityd,
    Rand3DElastic,
    SpatialCropd,
    Resize
)

from monai.data import (
    CacheDataset,
    SmartCacheDataset,
    load_decathlon_datalist,
    DataLoader,
    Dataset,
    DistributedSampler
)

import h5py
from custom_image_dataset import CustomImageDataset
from sklearn.preprocessing import OneHotEncoder
import torchio as tio

def get_loader(args):
    datadir = args.data_dir
    #jsonlist = os.path.join(datadir, args.json_list)
    jsonlist = args.json_list
    num_workers = args.num_workers
    
    #num_none_list = [669, 966, 1567]
    
    new_datalist = []
    datalist = load_decathlon_datalist(jsonlist, False, "training", base_dir=datadir)
    for item in datalist:
        item_name = ''.join(item['image']).split('.')[0].split('/')[-2]
        item_num = int(''.join(item_name).split('_')[1])
        #if item_num in num_none_list:
            #continue
        
        item_dict = {'image': item['image'], 'name': item_name}
        new_datalist.append(item_dict)
    
    new_vallist = []
    vallist = load_decathlon_datalist(jsonlist, False, "validation", base_dir=datadir)
    for item in vallist:
        item_name = ''.join(item['image']).split('.')[0].split('/')[-2]
        item_num = int(''.join(item_name).split('_')[1])
        #if item_num in num_none_list:
            #continue
        
        item_dict = {'image': item['image'], 'name': item_name}
        new_vallist.append(item_dict)
    
    datalist = new_datalist
    val_files = new_vallist
    
    print('Dataset all training: number of data: {}'.format(len(datalist)))
    print('Dataset all validation: number of data: {}'.format(len(val_files)))

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Orientationd(keys=["image"], axcodes="RAI", as_closest_canonical=True),
            ThresholdIntensityd(keys=["image"], 
                                threshold=args.a_max, 
                                above=False, 
                                cval=args.a_max, 
                                allow_missing_keys=False),
            ThresholdIntensityd(keys=["image"], 
                                threshold=args.a_min, 
                                above=True, 
                                cval=args.a_min, 
                                allow_missing_keys=False),
            
            ScaleIntensityRanged(keys=["image"],
                                 a_min=args.a_min,
                                 a_max=args.a_max,
                                 b_min=args.b_min,
                                 b_max=args.b_max,
                                 clip=True),
            # SpatialPadd(keys="image", spatial_size=[args.roi_x,
            #                                         args.roi_y,
            #                                         args.roi_z]),
            # CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x,
            #                                                                  args.roi_y,
            #                                                                  args.roi_z]),
            # RandSpatialCropSamplesd(
            #     keys=["image"],
            #     roi_size=[args.roi_x,
            #               args.roi_y,
            #               args.roi_z],
            #     num_samples=args.global_crops_number,
            #     random_center=True,
            #     random_size=False
            # ),

            ToTensord(keys=["image"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Orientationd(keys=["image"], axcodes="RAI", as_closest_canonical=True),
            ThresholdIntensityd(keys=["image"], 
                                threshold=args.a_max, 
                                above=False, 
                                cval=args.a_max, 
                                allow_missing_keys=False),
            ThresholdIntensityd(keys=["image"], 
                                threshold=args.a_min, 
                                above=True, 
                                cval=args.a_min, 
                                allow_missing_keys=False),
            ScaleIntensityRanged(keys=["image"],
                                 a_min=args.a_min,
                                 a_max=args.a_max,
                                 b_min=args.b_min,
                                 b_max=args.b_max,
                                 clip=True),
            # SpatialPadd(keys="image", spatial_size=[args.roi_x,
            #                                         args.roi_y,
            #                                         args.roi_z]),
            # CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x,
            #                                                                  args.roi_y,
            #                                                                  args.roi_z]),
            # RandSpatialCropSamplesd(
            #     keys=["image"],
            #     roi_size=[args.roi_x,
            #               args.roi_y,
            #               args.roi_z],
            #     num_samples=args.global_crops_number,
            #     random_center=True,
            #     random_size=False
            # ),
            ToTensord(keys=["image"]),
        ]
    )

    if args.normal_dataset:
        print('Using Normal dataset')
        dataset = Dataset(data=datalist, transform=train_transforms)

    elif args.smartcache_dataset:
        print('Using SmartCacheDataset')
        dataset = SmartCacheDataset(data=datalist,
                                    transform=train_transforms,
                                    replace_rate=1,
                                    cache_rate=0.1)

    else:
        print('Using MONAI Cache Dataset')
        dataset = CacheDataset(data=datalist,
                               transform=train_transforms,
                               cache_rate=1,
                               num_workers=num_workers)
    
    if args.distributed:
        train_sampler = DistributedSampler(dataset=dataset,
                                           even_divisible=True,
                                           shuffle=True)
    else:
        train_sampler = None
    
    train_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              num_workers=num_workers,
                              sampler=train_sampler,
                              drop_last=True)
    
    val_ds = SmartCacheDataset(data=val_files,
                               transform=val_transforms,
                               replace_rate=1,
                               cache_rate=0.1)

    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            drop_last=True)

    return train_loader, val_loader

def get_loader_adni_ukb(args, test=False):
    datadir = args.data_dir
    num_workers = args.num_workers
    
    train_transform = Compose(
        [
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.CropOrPad((128, 128, 128)),
        ]
    )

    val_transform = Compose(
        [
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.CropOrPad((128, 128, 128)),
        ]
    )

    if args.adni_dataset:
        print('Using ADNI dataset')
        train_ds = get_adni_trainset(args, train_transform=train_transform, test=test)
        val_ds = get_adni_valset(args, val_transform=val_transform, test=test)
    elif args.ukb_dataset:
        print('Using UKB dataset')
        train_ds = get_ukb_trainset(args, train_transform=train_transform, test=test)
        val_ds = get_ukb_valset(args, val_transform=val_transform, test=test)


    print('Dataset all training: number of data: {}'.format(len(train_ds)))
    print('Dataset all validation: number of data: {}'.format(len(val_ds)))
    
    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds,
                                           even_divisible=True,
                                           shuffle=True)
        val_sampler = DistributedSampler(dataset=val_ds,
                                        even_divisible=True,
                                        shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              num_workers=num_workers,
                              sampler=train_sampler,
                              drop_last=True)
    
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            num_workers=num_workers,
                            sampler=val_sampler,
                            shuffle=False,
                            drop_last=True)

    return train_loader, val_loader

def get_adni_trainset(args, train_transform=None, test=False):
    data_dir = os.path.join(args.data_dir, "train.h5")
    diagnosis = []
    image_train = []
    label_train = []
    i = 0
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if test and i>=16:
                break
            if name == "stats":
                continue
            rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            image_train.append(transformed_image)
            label_train.append(group.attrs['DX'])
            i += 1


    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_train = OH_encoder.transform(np.array(label_train).reshape(-1, 1))
    train_set = CustomImageDataset(image_train, label_train)
    return train_set

def get_adni_valset(args, val_transform=None, test=False):
    data_dir = os.path.join(args.data_dir, "valid.h5")
    diagnosis = []
    image_val = []
    label_val = []
    i = 0
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if test and i>=16:
                break
            if name == "stats":
                continue
            rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if val_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = val_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            image_val.append(transformed_image)
            label_val.append(group.attrs['DX'])
            i += 1

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_val = OH_encoder.transform(np.array(label_val).reshape(-1, 1))
    val_set = CustomImageDataset(image_val, label_val)
    return val_set

## UKB, no label?
def get_ukb_trainset(args, train_transform=None, test=False):
    data_dir = os.path.join(args.data_dir, "uk_train.h5")
    image_train = []
    label_train = []
    i = 0
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if test and i>=16:
                break
            if name == "stats":
                continue
            # rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            image_train.append(transformed_image)
            label_train.append('X')
            i += 1

    train_set = CustomImageDataset(image_train, label_train)
    return train_set

def get_ukb_valset(args, val_transform=None, test=False):
    data_dir = os.path.join(args.data_dir, "uk_valid.h5")
    image_val = []
    label_val = []
    i = 0
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if test and i>=16:
                break
            if name == "stats":
                continue
            # rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if val_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = val_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            image_val.append(transformed_image)
            label_val.append('X')
            i += 1
    val_set = CustomImageDataset(image_val, label_val)
    return val_set




def get_loader_hos_dzne(args, fold=1, mode="train"):
    datadir = args.data_dir
    num_workers = args.num_workers

    if args.resize:
        train_transform = Compose(
            [
                tio.RescaleIntensity(out_min_max=(0, 1)),
                tio.CropOrPad((128, 128, 128)),
                tio.Resize((args.roi_x, args.roi_y, args.roi_z)),
            ]
        )

    else:
        train_transform = Compose(
            [
                tio.RescaleIntensity(out_min_max=(0, 1)),
                tio.CropOrPad((128, 128, 128)),
            ]
        )

    if args.hos_dataset:
        print('Using hospital dataset')
        train_ds = get_hos_dataset(datadir, i=fold, mode=mode, train_transform=train_transform)
        print(f'Hospital {mode}set created')
    elif args.dzne_dataset:
        print('Using DZNE dataset')
        train_ds = get_dzne_dataset(datadir, i=fold, mode=mode, train_transform=train_transform)
        print(f'DZNE {mode}set created')


    print('Number of data: {}'.format(len(train_ds)))
    
    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds,
                                           even_divisible=True,
                                           shuffle=True)
    else:
        train_sampler = None
    
    if mode=="train":
        train_loader = DataLoader(train_ds,
                                batch_size=args.batch_size,
                                num_workers=num_workers,
                                sampler=train_sampler,
                                drop_last=True)
    else:
        train_loader = DataLoader(train_ds,
                                batch_size=args.batch_size,
                                num_workers=num_workers,
                                sampler=train_sampler,
                                shuffle=False,
                                drop_last=True)

    print(f'{mode}set data loader created')

    return train_loader


def get_hos_dataset(root_path, i=1, mode="train", train_transform=None):
    suffix = "238+19+72_tum.h5"
    data_dir = os.path.join(root_path, suffix)
    if mode=="train":
        train_data = np.load(f'{root_path}{i}-train.npy', allow_pickle=True)
    elif mode=="val":
        train_data = np.load(f'{root_path}{i}-valid.npy', allow_pickle=True)
    else:
        train_data = np.load(f'{root_path}{i}-test.npy', allow_pickle=True)
    diagnosis = []
    image_train = []
    label_train = []
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            if rid in train_data:
                image_train.append(transformed_image)
                label_train.append(group.attrs['DX'])

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_train = OH_encoder.transform(np.array(label_train).reshape(-1, 1))
    train_set = CustomImageDataset(image_train, label_train)
    return train_set

def get_dzne_dataset(root_path, i=1, mode="train", train_transform=None):
    suffix = "DZNE_CN_FTD_AD.h5"
    data_dir = os.path.join(root_path, suffix)
    if mode=="train":
        train_df = pd.read_csv(f'{root_path}{i}-train.csv')
        train_data = list(train_df["IMAGEID"])
    elif mode=="val":
        train_df = pd.read_csv(f'{root_path}{i}-valid.csv')
        train_data = list(train_df["IMAGEID"])
    else:
        train_df = pd.read_csv(f'{root_path}{i}-test.csv')
        train_data = list(train_df["IMAGEID"])

    diagnosis = []
    image_train = []
    label_train = []
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs["IMAGEID"]
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            if rid in train_data:
                image_train.append(transformed_image)
                label_train.append(group.attrs['DX'])


    # print(len(diagnosis))
    # print(len(label_train))
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_train = OH_encoder.transform(np.array(label_train).reshape(-1, 1))
    train_set = CustomImageDataset(image_train, label_train)
    return train_set