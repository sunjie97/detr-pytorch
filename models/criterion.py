import torch 
import torch.nn as nn 
import torch.nn.functional as F
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class Criterion(nn.Module):

    def __init__(self, matcher, weight_dict, eos_coef, num_classes):

        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher 
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef 

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def _get_match_idx(self, match_indices):
        # tensor([0, 0, 1, 1, 1, 1, 1, 1])  标志每个smaple的标签数
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        # tensor([1, 5, 0, 1, 2, 3, 4, 5])  所有的预测值
        src_idx = torch.cat([src for (src, _) in match_indices])
        tgt_idx = torch.cat([tgt for (_, tgt) in match_indices])

        return batch_idx, src_idx, tgt_idx

    def class_loss(self, preds, gt_labels, match_indices):
        # [bs, num_queries, num_classes+1]  加的1是空类别
        pred_logits = preds['pred_logits']
        batch_idx, src_idx, _ = self._get_match_idx(match_indices)
        gt_labels = torch.cat([gt_label[i] for gt_label, (_, i) in zip(gt_labels, match_indices)])
        # 初始化一个 shape 为 [bs, num_queries], 值为空类别的矩阵
        target_labels = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        target_labels[batch_idx, src_idx] = gt_labels 

        loss = F.cross_entropy(pred_logits.transpose(1, 2), target_labels, self.empty_weight.to(target_labels.device))

        return loss 

    def box_loss(self, preds, gt_boxes, match_indices):
        num_boxes = sum(len(gt_box) for gt_box in gt_boxes)
        batch_idx, src_idx, _ = self._get_match_idx(match_indices)

        pred_boxes = preds['pred_boxes']
        pred_boxes = pred_boxes[batch_idx, src_idx]
        target_boxes = torch.cat([gt_box[i] for gt_box, (_, i) in zip(gt_boxes, match_indices)])
        box_loss = F.l1_loss(pred_boxes, target_boxes, reduction='none')
        giou_loss = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes)
        ))

        return box_loss.sum() / num_boxes, giou_loss.sum() / num_boxes

    def forward(self, preds, targets):
        pred_without_aux = {k: v for k, v in preds.items() if k != 'aux_outs'}

        match_indices = self.matcher(pred_without_aux, targets)

        class_loss = self.class_loss(preds, targets['labels'], match_indices)
        box_loss = self.box_loss(preds, targets['boxes'], match_indices)

        loss_dict = {
            'class_loss': class_loss,
            'box_loss': box_loss[0],
            'giou_loss': box_loss[1]
        }

        if 'aux_outs' in preds:
            for i, aux_out in enumerate(preds['aux_outs']):
                match_indices = self.matcher(aux_out, targets)
                class_loss = self.class_loss(preds, targets['labels'], match_indices)
                box_loss = self.box_loss(preds, targets['boxes'], match_indices)
                aux_losses = {
                    f'class_loss_{i}': class_loss,
                    f'box_loss_{i}': box_loss[0],
                    f'giou_loss_{i}': box_loss[1]
                }
                loss_dict.update(aux_losses)

        return loss_dict 


def build_criterion(args):
    from .matcher import build_matcher

    matcher = build_matcher(args)

    weight_dict = {
        'class_loss': args.class_loss_coef, 
        'box_loss': args.box_loss_coef, 
        'giou_loss': args.giou_loss_coef
        }
    
    if not args.no_aux_loss:
        aux_weight_dict = {}
        for i in range(args.decoder_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    return Criterion(matcher, weight_dict, args.eos_coef, args.num_classes) 
