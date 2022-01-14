import os 
import torch
import cv2 
from collections import defaultdict
from pycocotools.coco import COCO

from utils.misc import NestedTensor 


class CocoDataset(torch.utils.data.Dataset):

    def __init__(self, root, ann_path, transforms=None):
        super().__init__()

        self.root = root 
        self.coco = COCO(ann_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        self.transforms = transforms 

    def _load_img(self, idx):
        file_name = self.coco.loadImgs(idx)[0]['file_name']
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, file_name)), cv2.COLOR_BGR2RGB)
        
        return img 

    def _load_target(self, idx):
        target = self.coco.loadAnns(self.coco.getAnnIds(idx))

        bboxes = [t['bbox'] for t in target]
        labels = [t['category_id'] for t in target]

        return bboxes, labels 

    def __getitem__(self, idx):
        idx = self.ids[idx]
        img = self._load_img(idx)
        bboxes, labels = self._load_target(idx)

        sample = {
            'img': img,
            'gt_bboxes': bboxes,
            'gt_labels': labels
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):

        return len(self.ids)


def coco_collate_fn(batch):
    """
    每个 batch 有多张图片，但每张图片尺寸并不一致，所以需要对同一个 batch 内的图片以最大尺寸做 padding, NestedTensor 里面的 mask 用于标识 padding 位置
    """
    data = defaultdict(list)
    for k in batch[0].keys():
        for b in batch:
            data[k].append(b[k])
    imgs = data['img']

    max_size = (0, 0, 0)
    img_sizes = [img.shape for img in imgs]
    for img_size in img_sizes:
        max_size = [max(max_size[i], img_size[i]) for i in range(len(img_sizes[0]))]

    batch_shape = [len(imgs)] + max_size 
    b, _, h, w = batch_shape 

    padded_imgs = torch.zeros(batch_shape, dtype=imgs[0].dtype, device=imgs[0].device)
    masks = torch.ones((b, h, w), dtype=torch.bool, device=imgs[0].device)

    for img, padded_img, mask in zip(imgs, padded_imgs, masks):
        padded_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        # 存在内容的区域为0，pad的区域为1
        mask[:img.shape[1], :img.shape[2]] = False 
    
    data['img'] = NestedTensor(padded_imgs, masks)

    return data
