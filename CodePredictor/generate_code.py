# Standard Library Imports
import argparse
import datetime
import io
import logging
import os
import random
import shutil
import sys
sys.path.append(os.path.dirname(__file__))
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

from dataset.vggsound_qmel import VGGSoundQmelDataset_test, code2emb_qmel
from models.transformer import CodePredictor
from encodec import EncodecModel


class VideoCodeGenerator():
    def __init__(self, args, ckpt_path):
        self.args = args        
        n_classifiers = args.num_win // 4
        n_labels = np.power(args.num_quan, 4)
        tseq_size = args.num_win
        us_rate = 4 
        self.us_rate = us_rate  
        self.n_classifiers = n_classifiers
        self.tseq_size = tseq_size
        input_size = 768
        self.model = CodePredictor(us_rate=us_rate, n_classifiers=n_classifiers, n_labels=n_labels, n_layers=6, input_size=input_size, tseq_size=tseq_size, embed_size=512, heads=8, dropout=0.1, forward_expansion=4, window_length=25, seq_length=250)
        
        ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
        new_dict = {}
        for k in ckpt.keys():
            new_dict[k[6:]] = ckpt[k]
        self.model.load_state_dict(new_dict)
        self.model.cuda().eval()

    
    def pred2emb(self, prediction, mean=None, std=None):
        emb = code2emb_qmel(prediction, num_quan=self.args.num_quan, num_win=self.args.num_win)
        return emb
    
    
    def generate(self, vfeat, effect_lengths=None, mean=None, std=None):
        if effect_lengths is None:
            assert(vfeat.shape[0] == 1)
            effect_lengths = [vfeat.shape[1]]
        num_gen = effect_lengths[0].item() * self.us_rate
        
        with torch.no_grad():
            enc_output = self.model.encode(vfeat, effect_lengths)

        with torch.no_grad():
            prediction, pred_ms = self.model.decode(enc_output, effect_lengths)
            pred_mean = pred_ms[:,:,0]
            pred_std = pred_ms[:,:,1]
            
            B, L, C = prediction[0].shape
            prediction_cat = torch.cat(prediction, dim=0).contiguous().view(self.n_classifiers, B, L, C).permute(1, 2, 0, 3)
            prediction_rs = torch.max(prediction_cat, dim=3)[1]
            tseqs = []
            for i in range(B):
                tseqs.append(self.pred2emb(prediction_rs[i]).unsqueeze(0))
                
            tseq = torch.cat(tseqs, dim=0)
            tseq = tseq * pred_std.unsqueeze(2) + pred_mean.unsqueeze(2) 
        
        return prediction, pred_mean, pred_std, tseq


    def inference(self, batch):
        effect_length, vfeat, video_index = batch
        effect_length = effect_length.cuda()
        vfeat = vfeat.cuda()
    
        prediction, pred_mean, pred_std, tseq = self.generate(vfeat, effect_length)
        
        effect_mask = torch.zeros_like(prediction[0][:, :, 0])
        for i in range(effect_mask.shape[0]):
            effect_mask[i, :effect_length[i]] = 1
        
        return {'prediction': prediction, 'effect_mask': effect_mask, 'emb_pred': tseq, 'pred_mean': pred_mean, 'pred_std': pred_std}


def args_parse():
    args = argparse.ArgumentParser()
    args.add_argument("--root", type=str, default='', help="root path")
    args.add_argument("--num_quan", type=int, default=3, help="number of qmel values")
    args.add_argument("--num_win", type=int, default=8, help="number of frequency windows")
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--num_workers", type=int, default=12)
    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = args_parse()
    root_path = args.root
    
    ckpt_path = os.path.join('pretrain', 'CodePredictor.ckpt')
    
    # Load test dataset
    test_dataset = VGGSoundQmelDataset_test(root_path + '/video_clip', num_quan=args.num_quan, num_win=args.num_win)
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, 
            num_workers=args.num_workers, drop_last=False)
    
    save_path = os.path.join(root_path, 'rst_emb')
    os.makedirs(save_path, exist_ok=True)
    save_path_mean = os.path.join(root_path, 'rst_mean')
    os.makedirs(save_path_mean, exist_ok=True)
    save_path_std = os.path.join(root_path, 'rst_std')
    os.makedirs(save_path_std, exist_ok=True)
    
    # Load model
    vcgen = VideoCodeGenerator(args, ckpt_path)
    
    # inference
    for batch in test_dataloader:
        video_indices = batch[-1]
        res = vcgen.inference(batch)
        emb_pred = res['emb_pred'].detach().cpu().numpy()
        mean_pred = res['pred_mean'].detach().cpu().numpy()
        std_pred = res['pred_std'].detach().cpu().numpy()
        for i in range(len(video_indices)):
            embi = emb_pred[i]
            video_index = video_indices[i]
            np.save(os.path.join(save_path, '{}.npy'.format(video_index)), embi)
            np.save(os.path.join(save_path_mean, '{}.npy'.format(video_index)), mean_pred[i])
            np.save(os.path.join(save_path_std, '{}.npy'.format(video_index)), std_pred[i])
        