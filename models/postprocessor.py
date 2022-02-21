import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from utils.box_ops import box_cxcywh_to_xyxy


class PostProcessor(nn.Module):

    @torch.no_grad()
    def forward(self, predictions, target_sizes, labelmap):

        pred_logits, pred_boxes = predictions['pred_logits'], predictions['pred_boxes']

        assert len(pred_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2 

        probs = F.softmax(pred_logits, -1)
        scores, labels_beforemap = probs[..., :-1].max(-1)
        labels = []
        for label in labels_beforemap:
            label = torch.tensor(
                [int(labelmap[str(l)]) for l in label.detach().cpu().numpy().tolist()], 
                dtype=labels_beforemap.dtype, 
                device=labels_beforemap.device
            )
            labels.append(label)

        # convert to [x0, y0, x1, y1] format 
        boxes = box_cxcywh_to_xyxy(pred_boxes)
        # from relative [0, 1] to absolute [0, height] coordinates
        img_w, img_h = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': score, 'labels': label, 'boxes': box} for score, label, box in zip(scores, labels, boxes)]

        return results 
