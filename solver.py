import os
import time 
import torch
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from models import build_detr
from datasets import build_loader
from models.criterion import build_criterion
from models.postprocessor import PostProcessor
from utils.misc import AverageMeter, create_logger, seed_everything
from utils.evaluator import Evaluator



class Solver:

    def __init__(self, args):
        self.args = args 
        seed_everything(self.args.seed)

        self.writer = SummaryWriter(args.save_dir + '/log')
        self.logger = create_logger(os.path.join(self.args.save_dir, f"log_{time.strftime('%Y%m%d_%H%M%S')}.txt"))
        self.device = torch.device(args.device)

        self.model = build_detr(args).to(self.device)
        self.train_loader, self.val_loader = build_loader(args)
        self.criterion = build_criterion(args)
        backbone_params = list(map(id, self.model.backbone.parameters()))
        other_params = filter(lambda p: id(p) not in backbone_params, self.model.parameters())
        params = [
            {'params': self.model.backbone.parameters(), 'lr': self.args.lr_backbone},
            {'params': other_params, 'lr': self.args.lr}
        ]
        self.optimizer = torch.optim.AdamW(params, weight_decay=self.args.weight_decay)
        self.postprocessor = PostProcessor()
        self.evaluator = Evaluator()
        self.scaler = GradScaler()

        coco91to80 = self.train_loader.dataset.coco91to80
        self.coco80to91 = dict([(str(v), k) for k, v in coco91to80.items()])

    def train(self):
        self.model.train()
        self.criterion.train()

        train_loss = AverageMeter()
        for batch_idx, (imgs, targets) in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            targets = dict((k, list(map(lambda x: x.cuda(), v))) for k, v in targets.items())

            with autocast():
                preds = self.model(imgs)
                loss_dict = self.criterion(preds, targets)
            losses = sum(loss_dict[k] * self.criterion.weight_dict[k] for k in loss_dict.keys())
            train_loss.update(losses.item())
            
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if (batch_idx + 1) % self.args.log_every == 0:
                self.logger.info(f'Training Iteration [{batch_idx+1}/{len(self.train_loader)}]: {train_loss.average:.3f}')
            
        return round(train_loss.average, 3)

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        # self.criterion.eval()
        for imgs, targets in self.val_loader:
            imgs = imgs.to(self.device)
            targets = dict((k, list(map(lambda x: x.cuda(), v))) for k, v in targets.items())

            preds = self.model(imgs)
            # loss_dict = self.criterion(preds, targets)
            # losses = sum(loss_dict[k] * self.criterion.weight_dict[k] for k in loss_dict.keys())
            
            orig_sizes = torch.stack(targets['orig_size'])
            results = self.postprocessor(preds, orig_sizes, self.coco80to91)
            results = {image_id.item(): output for image_id, output in zip(targets['image_id'], results)}

            self.evaluator.update(results)

        self.evaluator.summarize(self.val_loader.dataset.coco)
        self.evaluator.reset()

    def run(self):
        for epoch in range(self.args.num_epochs):
            train_loss = self.train()
            self.logger.info(f'Training Batch [{epoch+1}/{self.args.num_epochs}] {train_loss}')
            self.writer.add_scalar('train_loss', train_loss, global_step=epoch)

            self.eval()

