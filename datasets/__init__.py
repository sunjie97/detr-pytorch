import os
import torch 
from .transforms import Compose, RandomSelect, RandomHorizontalFlip, RandomResize, RandomSizeCrop, ToTensor, Normalize
from .dataset import CocoDataset, coco_collate_fn


def get_train_transforms():
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    return Compose([
        RandomHorizontalFlip(),
        RandomSelect(
            RandomResize(scales, max_size=1333),
            Compose([
                RandomResize([400, 500, 600]),
                RandomSizeCrop(384, 600),
                RandomResize(scales, max_size=1333)
            ])
        ),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



def get_test_transforms():
    return Compose([ 
        RandomResize([800], max_size=1333),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def build_loader(args):

    coco_path = args.coco_path

    train_root = os.path.join(coco_path, 'train2017')
    train_ann_path = os.path.join(coco_path, 'annotations', 'instances_train2017.json')
    val_root = os.path.join(coco_path, 'val2017')
    val_ann_path = os.path.join(coco_path, 'annotations', 'instances_val2017.json')

    train_dataset = CocoDataset(root=train_root, ann_path=train_ann_path, transforms=get_train_transforms())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, 
        pin_memory=True, drop_last=False, collate_fn=coco_collate_fn
    )

    val_dataset = CocoDataset(root=val_root, ann_path=val_ann_path, transforms=get_test_transforms(), phase='train')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False, 
        pin_memory=True, drop_last=False, collate_fn=coco_collate_fn
    )

    return train_loader, val_loader



