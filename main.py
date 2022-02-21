import argparse
from solver import Solver


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--log_every', default=1, type=int)

    # Model Parameters
    # * Backbone 
    parser.add_argument('--backbone', default='resnet50', type=str, 
                        choices=['resnet50', 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                                 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'])
    parser.add_argument('--position_embedding', default='sine', type=str, choices=['sine', 'learned'],
                        help='Type of positional embedding to use on top of the image features')

    # * Transformer
    parser.add_argument('--embed_dim', default=256, type=int, help='Size of the embeddings (dimension of the transformer)')
    parser.add_argument('--hidden_dim', default=2048, type=int, 
                        help='Intermediate size of the feedforward layers in the transformer block')
    parser.add_argument('--nheads', default=8, type=int, help='Number of attention heads inside the transformer\'s attentions')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout applied in the transformer')
    parser.add_argument('--encoder_layers', default=6, type=int, help='Number of encoder layers in the transformer')
    parser.add_argument('--decoder_layers', default=6, type=int, help='Number of decoder layers in the transformer')
    parser.add_argument('--num_queries', default=100, type=int, help='Number of query slots')

    # Loss 
    parser.add_argument('--no_aux_loss', action='store_true', help='Disables auxiliary decoding losses (loss at each layer)')

    # * Matcher 
    parser.add_argument('--class_cost', default=1., type=float,
                        help='Class coefficient in the matching cost')
    parser.add_argument('--box_cost', default=5., type=float, 
                        help='L1 box coefficient in the matching cost')
    parser.add_argument('--giou_cost', default=2., type=float, 
                        help='GIoU bbox coefficient in the matching cost')

    # * Loss coefficients
    parser.add_argument('--class_loss_coef', default=1., type=float)
    parser.add_argument('--box_loss_coef', default=5., type=float)
    parser.add_argument('--giou_loss_coef', default=2., type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help='Relative classification weight of the no-object class')
    
    # dataset parameters
    parser.add_argument('--coco_path', default='/home/jie/Python/data/detr/model/data/coco', type=str)
    parser.add_argument('--num_classes', default=80, type=int)

    # training parameters
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_dir', default='outs', help='Path to save checkpoints and logs')

    args = parser.parse_args()

    return args 


def main(args):
    solver = Solver(args)

    solver.run()


if __name__ == '__main__':
    args = get_args()

    main(args)
