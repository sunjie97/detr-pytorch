import os
import json 
import torch
from PIL import Image 
from collections import defaultdict
from pycocotools.coco import COCO

from utils.misc import NestedTensor 



class CocoDataset(torch.utils.data.Dataset):

    def __init__(self, root, ann_path, transforms=None, phase='train'):
        super().__init__()

        self.root = root 
        self.phase = phase 
        self.transforms = transforms 
        self.coco = COCO(ann_path)

        ids = list(sorted(self.coco.imgs.keys()))
        if self.phase == "train":
            # 移除没有目标，或者目标面积非常小的数据
            valid_ids = self._coco_remove_images_without_annotations(ids)
            self.ids = valid_ids
        else:
            self.ids = ids

        self._init_label_map()
        
    def load_img(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, file_name)).convert("RGB")
        
        return img 

    def load_target(self, image_id, w, h):
        annos = self.coco.loadAnns(self.coco.getAnnIds(image_id))

        annos = [anno for anno in annos if anno['iscrowd'] == 0]

        # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
        boxes = []
        for anno in annos:
            if anno['bbox'][2] > 0 and anno['bbox'][3] > 0:
                boxes.append(anno['bbox'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # xywh -> xyxy
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(0, w)
        boxes[:, 1::2].clamp_(0, h)
        labels = [self.coco91to80[str(anno['category_id'])] for anno in annos]
        labels = torch.tensor(labels, dtype=torch.int64)

        targets = {}
        targets['boxes'] = boxes 
        targets['labels'] = labels 
        targets['image_id'] = torch.tensor([image_id])
        targets['orig_size'] = torch.tensor([w, h])

        # for conversion to coco api
        area = torch.tensor([anno["area"] for anno in annos])
        iscrowd = torch.tensor([anno["iscrowd"] for anno in annos])
        targets["area"] = area
        targets["iscrowd"] = iscrowd
        
        return targets 

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = self.load_img(image_id)
        targets = self.load_target(image_id, img.size[0], img.size[1])

        if self.transforms is not None:
            sample = self.transforms(img, targets)

        return sample

    def __len__(self):

        return len(self.ids)

    def _init_label_map(self):
        if self.phase == "train":
            # 获取coco数据索引与类别名称的关系
            # 注意在object80中的索引并不是连续的，虽然只有80个类别，但索引还是按照stuff91来排序的
            coco_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])

            # 将stuff91的类别索引重新编排，从1到80
            coco91to80 = dict([(str(k), idx+1) for idx, (k, _) in enumerate(coco_classes.items())])
            json_str = json.dumps(coco91to80, indent=4)
            with open('coco91_to_80.json', 'w') as json_file:
                json_file.write(json_str)

            # 记录重新编排后的索引以及类别名称关系
            coco80_info = dict([(str(idx+1), v) for idx, (_, v) in enumerate(coco_classes.items())])
            json_str = json.dumps(coco80_info, indent=4)
            with open('coco80_indices.json', 'w') as json_file:
                json_file.write(json_str)
        else:
            # 如果是验证集就直接读取生成好的数据
            coco91to80_path = 'coco91_to_80.json'
            assert os.path.exists(coco91to80_path), "file '{}' does not exist.".format(coco91to80_path)

            coco91to80 = json.load(open(coco91to80_path, "r"))

        self.coco91to80 = coco91to80

    def _coco_remove_images_without_annotations(self, ids):
        """
        删除coco数据集中没有目标，或者目标面积非常小的数据
        refer to:
        https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
        """
        def _has_only_empty_bbox(anno):
            return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

        def _has_valid_annotation(anno):
            # if it's empty, there is no annotation
            if len(anno) == 0:
                return False
            # if all boxes have close to zero area, there is no annotation
            if _has_only_empty_bbox(anno):
                return False

            return True

        valid_ids = []
        for img_id in ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)

            if _has_valid_annotation(anno):
                valid_ids.append(img_id)

        return valid_ids


def coco_collate_fn(batch):
    batch = list(zip(*batch))

    imgs = batch[0]
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
    
    imgs = NestedTensor(padded_imgs, masks)

    targets = defaultdict(list)
    for t in batch[1]:
        for k, v in t.items():
            targets[k].append(v)

    return imgs, targets
