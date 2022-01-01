import torch.nn as nn 
import torch.nn.functional as F

from .resnet import ResNet50
from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from utils.misc import NestedTensor


class Backbone(nn.Module):

    def __init__(self, backbone, position_embedding):
        super().__init__()

        self.backbone = backbone 
        self.position_embedding = position_embedding

    def forward(self, inputs):
        xs = self.backbone(inputs.tensors)
        
        outs = []
        for x in xs:
            m = inputs.mask 
            assert m is not None 
            mask = F.interpolate(m[None, ...].float(), size=x.shape[-2:]).to(torch.bool)[0]
            outs.append(NestedTensor(x, mask))

        pos = [self.position_embedding(o).to(o.tensors.dtype) for o in outs]
        return outs, pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2 
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(num_pos_feats=N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(num_pos_feats=N_steps)
    else:
        raise ValueError(f'not supported {args.position_embedding}')

    return position_embedding


def build_backbone(args):
    backbone = ResNet50()
    position_encoding = build_position_encoding(args)
    net = Backbone(backbone, position_encoding)

    return net 
