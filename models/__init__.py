from .backbone import build_backbone
from .transformer import build_transformer
from .detr import DETR


def build_detr(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = DETR(backbone, transformer, num_classes=args.num_classes, num_queries=args.num_queries, aux_loss=not args.no_aux_loss)

    return model 
