import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou



class HungarianMatcher(nn.Module):
    """ This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects)
    """
    def __init__(self, class_cost=1., bbox_cost=1., giou_cost=1.):
        super().__init__()

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

        assert class_cost != 0 or bbox_cost != 0 or giou_cost != 0, "all cost can't be 0"

    @torch.no_grad()
    def forward(self, preds, targets):
        """ Performs the matching 
        
        Args:
            preds: This is a dict that contains at least these entries:
                pred_logits: Tensor of dim [bs, num_queries, num_classes] with the classification logits 
                pred_boxes:  Tensor of dim [bs, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targtes (len(targets) = batch_size), where each target is a dict containing:
                labels: Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth objects in the target)
                    containing the class labels 
                boxes: Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                index_i is the indices of the selected predictions (in order)
                index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        """
        targets:
            [{labels: torch.size([num_target_boxes]), boxes: torch.Size([num_target_boxes, 4])}, ...]

            其中 len(targets) = batch_size
        
        """
        
        bs, num_queries, _ = preds['pred_logits']  # [bs, num_queries, num_classes+1]

        pred_logits = preds['pred_logits'].flatten(0, 1).softmax(-1)  # [bs * num_queries, num_classes+1]
        pred_boxes = preds['pred_boxes'].flatten(0, 1)  # [bs * num_queries, 4]

        target_labels = torch.cat([v['labels'] for v in targets])  # [num_target_boxes]
        target_boxes = torch.cat([v['boxes'] for v in targets])  # [num_target_boxes, 4]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class]
        # The 1 is a constant that doesn't change the matching, it can be ommitted
        class_cost = -pred_logits[:, target_labels]  # [bs*num_queries, num_target_boxes]

        # Compute the L1 cost between boxes 
        bbox_cost = torch.cdist(pred_boxes, target_boxes, p=1)  # [bs * num_queries]

        # Compute the giou cost between boxes 
        giou_cost = -generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes))

        # Final cost matrix 
        C = self.class_cost * class_cost + self.bbox_cost * bbox_cost + self.giou_cost * giou_cost
        C = C.view(bs, num_queries, -1).cpu()  # [bs, num_queries, num_target_boxes]

        sizes = [len(v['labels']) for v in targets]  # targets中包含batch_size个标注，sizes是一个列表，指明这个batch中每个图像中分别有多少标注
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):

    return HungarianMatcher(class_cost=args.class_cost, bbox_cost=args.bbox_cost, giou_cost=args.giou_cost)
