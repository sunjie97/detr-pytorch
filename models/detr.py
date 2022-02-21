import torch
import torch.nn as nn 
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()

        self.num_layers = num_layers 
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([nn.Linear(n, k) for n, k in zip([in_dim] + h, h + [out_dim])])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x 


class DETR(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):

        super().__init__()

        self.backbone = backbone 
        self.transformer = transformer
        self.num_queries = num_queries
        self.aux_loss = aux_loss

        embed_dim = transformer.embed_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, embed_dim, kernel_size=1)
        self.class_embed = nn.Linear(embed_dim, num_classes+1)
        self.box_embed = MLP(embed_dim, embed_dim, out_dim=4, num_layers=3)
        self.query_embed = nn.Embedding(num_queries, embed_dim)

    def forward(self, samples):

        feature, pos_embed = self.backbone(samples)

        src, mask = feature.decompose()
        assert mask is not None 

        # hs: [num_layers, b, num_queries, embed_dim] 
        hs, _ = self.transformer(self.input_proj(src), key_padding_mask=mask, query_embed=self.query_embed.weight, pos_embed=pos_embed)

        out_logits = self.class_embed(hs)
        out_boxes = self.box_embed(hs).sigmoid()
        out = {
            'pred_logits': out_logits[-1],
            'pred_boxes': out_boxes[-1]
        }

        if self.aux_loss:
            out['aux_outs'] = self._set_aux_loss(out_logits, out_boxes)

        return out 

    @torch.jit.unused
    def _set_aux_loss(self, out_logits, out_boxes):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(out_logits[:-1], out_boxes[:-1])]
