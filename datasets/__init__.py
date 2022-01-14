import torch 
from torchvision.transforms import Compose
from .dataset import CocoDataset, coco_collate_fn
from .transforms import Resize, RandomCrop, AutoAugment, RandomFlip, Normalize, Pad, ToTensor


auto_augment_transforms = [
    [
        Resize(img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), 
                          (736, 1333), (768, 1333), (800, 1333)], multiscale_mode='value', keep_ratio=True),
    ],
    [
        Resize(img_scale=[(400, 1333), (500, 1333), (600, 1333)], multiscale_mode='value', keep_ratio=True),
        RandomCrop(crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
        Resize(img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), 
                          (736, 1333), (768, 1333), (800, 1333)], multiscale_mode='value', override=True, keep_ratio=True),
    ]
]


def get_train_transforms():
    transforms = Compose([
        RandomFlip(flip_ratio=0.5),
        AutoAugment(transforms=auto_augment_transforms),
        Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        Pad(size_divisor=32),
        ToTensor()
    ])
    return transforms


def build_loader(cfg):

    dataset = CocoDataset(root=cfg['root'], ann_path=cfg['ann_path'], transforms=get_train_transforms())

    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=4, shuffle=True, pin_memory=True, drop_last=False, collate_fn=coco_collate_fn)

    return loader 



