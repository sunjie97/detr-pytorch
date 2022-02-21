import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou



class HungarianMatcher(nn.Module):

    def __init__(self, class_cost=1., box_cost=1., giou_cost=1.):
        super().__init__()

        self.class_cost = class_cost
        self.box_cost = box_cost
        self.giou_cost = giou_cost

        assert class_cost != 0 or box_cost != 0 or giou_cost != 0, "all cost can't be 0"

    @torch.no_grad()
    def forward(self, preds, targets):

        bs, num_queries, _ = preds['pred_logits'].shape  # [bs, num_queries, num_classes+1]
        # targets中包含batch_size个标注，sizes是一个列表，指明这个batch中每个图像中分别有多少标注
        sizes = [len(gt_label) for gt_label in targets['labels']]  

        pred_logits = preds['pred_logits'].flatten(0, 1).softmax(-1)  # [bs * num_queries, num_classes+1]
        pred_boxes = preds['pred_boxes'].flatten(0, 1)  # [bs * num_queries, 4]

        gt_labels = torch.cat(targets['labels'])
        gt_boxes = torch.cat(targets['boxes'])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[gt class]
        # The 1 is a constant that doesn't change the matching, it can be ommitted
        class_cost = -pred_logits[:, gt_labels]  # [bs * num_queries, num_gt_boxes]
        # Compute the L1 cost between boxes 
        box_cost = torch.cdist(pred_boxes, gt_boxes, p=1)  # [bs * num_queries]
        # Compute the giou cost between boxes 
        giou_cost = -generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))

        # Final cost matrix 
        C = self.class_cost * class_cost + self.box_cost * box_cost + self.giou_cost * giou_cost
        C = C.view(bs, num_queries, -1).cpu()  # [bs, num_queries, num_gt_boxes]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):

    return HungarianMatcher(class_cost=args.class_cost, box_cost=args.box_cost, giou_cost=args.giou_cost)
