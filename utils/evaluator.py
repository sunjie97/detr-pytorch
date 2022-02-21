import json 
import torch
import numpy as np
from pycocotools.cocoeval import COCOeval


class Evaluator:

    def __init__(self):
        self.coco_pred = []
        self.img_ids = []

    def update(self, preds):
        img_ids = list(np.unique(list(preds.keys())))
        self.img_ids.extend(img_ids)

        for img_id, pred in preds.items():
            if len(pred) == 0:
                continue

            boxes = pred['boxes']
            xmin, ymin, xmax, ymax = boxes.unbind(1)
            boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1).tolist()
            scores = pred['scores'].tolist()
            labels = pred['labels'].tolist()

            self.coco_pred.extend([
                {
                    'image_id': img_id,
                    'bbox': boxes[i],
                    'category_id': labels[i],
                    'score': scores[i]
                }
                for i in range(len(labels))
            ])

    def summarize(self, coco_true):
        
        json_str = json.dumps(self.coco_pred, indent=4)
        with open('predict_tmp.json', 'w') as json_file:
            json_file.write(json_str)

        coco_pre = coco_true.loadRes('predict_tmp.json')

        coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    def reset(self):
        self.coco_pred = []
        self.img_ids = []


