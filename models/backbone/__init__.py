import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .resnet import ResNet50
from .mobilenetv2 import MobilenetV2
from .mobilenetv3 import MobileNetV3Small, MobileNetV3Large
from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from utils.misc import NestedTensor


model_urls = {
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
}


def build_backbone(args):
    if args.backbone == 'resnet50':
        backbone = ResNet50()
    elif args.backbone == 'mobilenet_v2':
        backbone = MobilenetV2()
    elif args.backbone == 'mobilenet_v3_small':
        backbone = MobileNetV3Small()
    elif args.backbone == 'mobilenet_v3_large':
        backbone = MobileNetV3Large()
    state_dict = torch.hub.load_state_dict_from_url(model_urls[args.backbone])
    for k in list(state_dict.keys()):
        if 'fc' in k or 'classifier' in k:
            del state_dict[k]
    backbone.load_state_dict(state_dict)

    position_encoding = build_position_encoding(args)
    net = Backbone(backbone, position_encoding)

    return net 


def build_position_encoding(args):
    N_steps = args.embed_dim // 2 
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(num_pos_feats=N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(num_pos_feats=N_steps)
    else:
        raise ValueError(f'not supported {args.position_embedding}')

    return position_embedding


class Backbone(nn.Module):

    def __init__(self, backbone, position_embedding):
        super().__init__()

        self.backbone = backbone 
        self.position_embedding = position_embedding

        self.num_channels = backbone.num_channels

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
