# Standard Library Imports
import argparse
import datetime
import io
import logging
import os
import random
import shutil
import sys
import time
import psutil
# Third-Party Library Imports
import cv2 as cv
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim, Tensor
from torch.autograd import Function, Variable
from torch.nn import functional as F
import torch.nn as nn
import torch
import torchvision
import transformers
from torch.utils.data import (DataLoader, RandomSampler, Subset, DistributedSampler)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

# Local application/library specific imports
from dataset.vggsound_qmel import VGGSoundQmelDataset
from models.transformer import CodePredictor

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduce = reduce
    
    def forward(self, pred, target):
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss


class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dataset = VGGSoundQmelDataset('train', args.root_path, num_quan=args.num_quan, num_win=args.num_win)
        self.test_dataset = VGGSoundQmelDataset('test', args.root_path, num_quan=args.num_quan, num_win=args.num_win)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, 
            num_workers=self.args.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, 
            num_workers=self.args.num_workers, drop_last=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, 
            num_workers=self.args.num_workers, drop_last=False)



class VideoCodePredictor(pl.LightningModule):
    def __init__(self, args, num_training_steps):
        super(VideoCodePredictor, self).__init__()
        self.args = args
        self.num_training_steps = num_training_steps
        
        n_classifiers = args.num_win // 4
        n_labels = np.power(args.num_quan, 4)
        tseq_size = args.num_win
        us_rate = 4
        input_size = 768
        self.model = CodePredictor(us_rate=us_rate, n_classifiers=n_classifiers, n_labels=n_labels, n_layers=6, input_size=input_size, tseq_size=tseq_size, embed_size=512, heads=8, dropout=0.1, forward_expansion=4, window_length=25, seq_length=250)
        
        self.criterion = FocalLoss(gamma=2, reduce=False)

        self.criterion_ms = nn.MSELoss(reduce=False)
        self.Metrics = {'Loss': [], 'Acc': [], 'Loss_pred': [], 'Loss_mean': [], 'Loss_std': [], 'Nsample': []}
        self.BestMetrics = {'Loss': 1000, 'Acc': 0, 'Loss_pred': 1000, 'Loss_mean': 1000, 'Loss_std': 1000, 'Epoch': 0}
    
    def forward(self, vfeat, effect_lengths, tseq=None):
        return self.model(vfeat, effect_lengths, tseq)
 
    def training_step(self, batch, batch_idx):
        res = self.get_pred_shared_step(batch, batch_idx)
        return {'loss': res['loss'], 'loss_pred': res['loss_pred'], 'loss_mean': res['loss_mean'], 'loss_std': res['loss_std']}
        
    def validation_step(self, batch, batch_idx):
        self.get_valid_shared_step(batch, batch_idx)
    
    def compute_warmup(self, num_training_steps, num_warmup_steps):
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=-1,
            num_warmup_steps=self.args.num_warmup_steps,
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def get_pred_shared_step(self, batch, batch_idx):
        effect_length, vfeat, mean, std, emb, code = batch
        prediction, pred_ms = self(vfeat, effect_length)
        
        loss_pred = 0
        assert len(prediction) == code.shape[2], 'Predict and Target length: {}, {}'.format(len(prediction), code.shape[2])
        B, L, C = prediction[0].shape
        effect_mask = torch.zeros_like(prediction[0][:, :, 0])
        for i in range(B):
            effect_mask[i, :effect_length[i]] = 1
        effect_mask_flatten = effect_mask.contiguous().view(B * L)
        
        for i in range(len(prediction)):
            loss_pred += torch.sum(self.criterion(prediction[i].contiguous().view(B * L, C), code[:, :, i].view(B * L)) * effect_mask_flatten) / torch.sum(effect_mask_flatten) * self.args.pred_weight / len(prediction)

        pred_mean = pred_ms[:, :, 0]
        pred_std = pred_ms[:, :, 1]
        loss_mean = torch.sum(self.criterion_ms(pred_mean, mean).contiguous().view(B * L) * effect_mask_flatten) / torch.sum(effect_mask_flatten)
        loss_std = torch.sum(self.criterion_ms(pred_std, std).contiguous().view(B * L) * effect_mask_flatten) / torch.sum(effect_mask_flatten)
        
        loss = loss_pred + loss_mean + loss_std
        return {'loss': loss, 'prediction': prediction, 'pred_ms': pred_ms, 'target': code, 'effect_mask': effect_mask, 'loss_pred': loss_pred, 'loss_mean': loss_mean, 'loss_std': loss_std}

    def get_valid_shared_step(self, batch, batch_idx):
        res = self.get_pred_shared_step(batch, batch_idx)
        temp_metric = self.compute_Metric(res)
        for k in temp_metric.keys():
            self.Metrics[k].append(temp_metric[k])
    
    def compute_Metric(self, res):
        prediction = res['prediction']
        tcls = res['target']
        effect_mask = res['effect_mask'].unsqueeze(2).repeat(1, 1, tcls.shape[2])
        for i in range(len(prediction)):
            prediction[i] = prediction[i].unsqueeze(0)
        prediction = torch.cat(prediction, dim=0)
        pcls = torch.max(prediction, dim=3)[1].permute(1, 2, 0)
        assert pcls.shape == tcls.shape
        acc = torch.sum((pcls == tcls).float() * effect_mask) / torch.sum(effect_mask)
        metrics = {'Loss': res['loss'].detach().cpu().numpy(), 'Acc': acc.detach().cpu().numpy(), 'Nsample':torch.sum(effect_mask).detach().cpu().numpy()}
        metrics['Loss_pred'] = res['loss_pred'].detach().cpu().numpy()
        metrics['Loss_mean'] = res['loss_mean'].detach().cpu().numpy()
        metrics['Loss_std'] = res['loss_std'].detach().cpu().numpy()
        return metrics
              
    def get_valid_shared_end_step(self):
        loss, acc = round(np.sum(np.array(self.Metrics['Loss']) * np.array(self.Metrics['Nsample'])) / np.sum(self.Metrics['Nsample']), 4), round(np.sum(np.array(self.Metrics['Acc']) * np.array(self.Metrics['Nsample'])) / np.sum(self.Metrics['Nsample']) * 100, 2)
        self.log(f'Val@Loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'Val@Acc', acc, prog_bar=True, sync_dist=True)
        
        loss_pred = round(np.sum(np.array(self.Metrics['Loss_pred']) * np.array(self.Metrics['Nsample'])) / np.sum(self.Metrics['Nsample']), 4)
        loss_mean = round(np.sum(np.array(self.Metrics['Loss_mean']) * np.array(self.Metrics['Nsample'])) / np.sum(self.Metrics['Nsample']), 4)
        loss_std = round(np.sum(np.array(self.Metrics['Loss_std']) * np.array(self.Metrics['Nsample'])) / np.sum(self.Metrics['Nsample']), 4)
        self.log(f'Val@Loss_pred', loss_pred, prog_bar=True, sync_dist=True)
        self.log(f'Val@Loss_mean', loss_mean, prog_bar=True, sync_dist=True)
        self.log(f'Val@Loss_std', loss_std, prog_bar=True, sync_dist=True)
        if loss <= self.BestMetrics['Loss']:
            self.BestMetrics = {'Loss': loss, 'Acc': acc, 'Loss_pred': loss_pred, 'Loss_mean': loss_mean, 'Loss_std': loss_std, 'Epoch': self.current_epoch}
            print(self.BestMetrics)
        self.log(f'Best@Acc', self.BestMetrics['Acc'], prog_bar=True, sync_dist=True)
        self.log(f'Best@Loss', self.BestMetrics['Loss'], prog_bar=True, sync_dist=True)
        self.log(f'Best@Loss_pred', self.BestMetrics['Loss_pred'], prog_bar=True, sync_dist=True)
        self.log(f'Best@Loss_mean', self.BestMetrics['Loss_mean'], prog_bar=True, sync_dist=True)
        self.log(f'Best@Loss_std', self.BestMetrics['Loss_std'], prog_bar=True, sync_dist=True)
        self.log(f'Best@Epoch', self.BestMetrics['Epoch'], prog_bar=True, sync_dist=True)
        self.Metrics = {'Loss': [], 'Acc': [], 'Loss_pred': [], 'Loss_mean': [], 'Loss_std': [], 'Nsample': []}  

    def on_validation_epoch_end(self):
        self.get_valid_shared_end_step()
        

def args_parse():
    args = argparse.ArgumentParser()
    args.add_argument("--root_path", type=str, default='', help="root path")
    args.add_argument("--num_quan", type=int, default=3, help="number of qmel values")
    args.add_argument("--num_win", type=int, default=8, help="number of frequency windows")
    args.add_argument("--num_warmup_steps", type=float, default=0.1, help="ratio of warmup steps")
    args.add_argument("--lr", type=float, default=1e-4, help="learning rate for AdamW optimizer")
    # Args for experiments
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--pred_weight", type=float, default=1.0)
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--num_workers", type=int, default=12)
    args.add_argument("--log_dir", type=str, default="./Logs")
    args.add_argument("--device_id", type=int, default=0)
    args.add_argument("--val_freq", type=int, default=4)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--num_device", type=int, default=1)
    args.add_argument("--clip_max_norm", type=int, default=1)
    args = args.parse_args()
    return args
    

if __name__ == '__main__':
    args = args_parse()
    p = psutil.Process()
    p.cpu_affinity(list(np.arange(args.device_id, args.device_id+12)))
    
    # Adjust learning rate using batch size
    args.lr = args.lr * args.batch_size * args.num_device / 256
    
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir, 
        filename='{epoch}-{step}',     
        save_top_k=-1,            
        mode='max',
        every_n_epochs=1,
    )
    
    PLData = MyDataModule(args)
    num_training_steps = args.epochs * int(PLData.train_dataset.__len__() / (args.batch_size * args.num_device))
    PLModule = VideoCodePredictor(args, num_training_steps)

    logger = TensorBoardLogger(save_dir=log_dir, name="TBLogs")
    pl.seed_everything(args.seed)
    trainer = pl.Trainer(
        logger=logger,
        check_val_every_n_epoch=args.val_freq,
        gradient_clip_val=args.clip_max_norm, 
        gradient_clip_algorithm='norm',
        accelerator='gpu',
        max_epochs=args.epochs, 
        devices=args.num_device, 
        callbacks=[checkpoint_callback]) 
        
    trainer.fit(PLModule, PLData)