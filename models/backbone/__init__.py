import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .resnet import ResNet50
from .mobilenetv2 import MobilenetV2
from .mobilenetv3 import MobileNetV3Small, MobileNetV3Large
from .convnext import ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from utils.misc import NestedTensor


model_urls = {
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
    "convnext_tiny": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth"
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
    elif args.backbone == 'convnext_tiny':
        backbone = ConvNeXtTiny()
    elif args.backbone == 'convnext_small':
        backbone = ConvNeXtSmall()
    elif args.backbone == 'convnext_base':
        backbone = ConvNeXtBase()
    elif args.backbone == 'convnext_large':
        backbone = ConvNeXtLarge()
    elif args.backbone == 'convnext_xlarge':
        backbone = ConvNeXtXLarge()

    state_dict = torch.hub.load_state_dict_from_url(model_urls[args.backbone])
    if args.backbone.startswith('convnext'):
        state_dict = state_dict['model']
        del state_dict['norm.weight']
        del state_dict['norm.bias']
        del state_dict['head.weight']
        del state_dict['head.bias']
    else:
        for k in list(state_dict.keys()):
            if k.startswith('classifier') or k.startswith('fc'):
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
        x = self.backbone(inputs.tensors)
        
        m = inputs.mask 
        assert m is not None 
        mask = F.interpolate(m[None, ...].float(), size=x.shape[-2:]).to(torch.bool)[0]
        x = NestedTensor(x, mask)
        pos = self.position_embedding(x).to(x.tensors.dtype)

        return x, pos 
