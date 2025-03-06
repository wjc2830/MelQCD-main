import os
import sys
import cv2
import json
import yaml
import torch
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel
from torch.nn import functional as F
from moviepy.editor import VideoFileClip
from utils import instantiate_from_config

def get_video_features(frames):
    num_frames = len(frames)
    frames = torch.cat([frames[0].unsqueeze(0).repeat(8, 1, 1, 1), frames, frames[-1].unsqueeze(0).repeat(7, 1, 1, 1)], axis=0)
    segment_indices = (torch.arange(16).unsqueeze(0) + torch.arange(num_frames).unsqueeze(1)).cuda()
    segments = frames[segment_indices]
    segments = segments.permute(0, 2, 1, 3, 4).unsqueeze(0)
    rst, _ = model_sync(segments, for_loop=True)
    return rst[0]

def process_video(mp4_path):
    video_clip = VideoFileClip(mp4_path)
    frames = np.array([frame for frame in video_clip.iter_frames()])
    fps = video_clip.fps
    video_duration = video_clip.duration
    if fps != 25:
        resampled_indices = np.linspace(0, len(frames)-1, int(video_duration * 25), dtype=int)
        frames = frames[resampled_indices]
    
    frames_pad = []
    for i in range(len(frames)):
        frame = frames[i]
        h, w, _ = frame.shape
        padl = max(h, w)
        frame_pad = np.pad(frame, (((padl-h)//2, (padl-h) - (padl-h)//2), ((padl-w)//2, (padl-w) - (padl-w)//2), (0,0)), 'constant') 
        assert frame_pad.shape[0] == padl
        assert frame_pad.shape[1] == padl
        frames_pad.append(frame_pad)
    
    return frames_pad

def process_clip(frames_pad):
    
    frames_p = processor_clip(np.array(frames_pad), return_tensors='pt').to('cuda')
    with torch.no_grad():
        rst_clip = model_clip.get_image_features(**frames_p).cpu().numpy()
    return rst_clip

def process_sync(frames_pad):
    
    for i in range(256 - len(frames_pad)):
        frames_pad.append(np.zeros_like(frames_pad[-1]))
        
    frames_p = processor_sync(np.array(frames_pad), return_tensors='pt').to('cuda')
    with torch.no_grad():
        rst_sync = get_video_features(frames_p['pixel_values']).cpu().numpy()

    return rst_sync

def process(args):
    tar_root = args.root
    video_data_root = tar_root + '/video_clip'

    os.makedirs(tar_root, exist_ok=True)

    video_paths = sorted(glob(video_data_root + '/*.mp4'))

    os.makedirs(tar_root+'/syncformer/', exist_ok=True)
    os.makedirs(tar_root+'/CLIP_LargeEmbed/', exist_ok=True)
    
    for i, video_path in tqdm(enumerate(video_paths), desc=f'{len(video_paths)}'):
        try:
            base_name = os.path.basename(video_path)[:-4]
            if os.path.exists(os.path.join(tar_root+'/syncformer/', f"{base_name}.npy")):
                continue
            if os.path.exists(os.path.join(tar_root+'/CLIP_LargeEmbed/', f"{base_name}.npy")):
                continue
            
            videos = process_video(video_path)
            vfeat_clip = process_clip(videos)
            vfeat_sync = process_sync(videos)

            np.save(os.path.join(tar_root+'/CLIP_LargeEmbed/', f"{base_name}.npy"), vfeat_clip)
            np.save(os.path.join(tar_root+'/syncformer/', f"{base_name}.npy"), vfeat_sync)

        except Exception as e:
            print(e)

def args_parse():
    args = argparse.ArgumentParser()
    args.add_argument("--root", type=str, default='', help="root path")
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = args_parse() 
    # Load the CLIP model and processor
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor_clip = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model_clip.cuda().eval()

    processor_sync = CLIPImageProcessor(image_mean=0.5, image_std=0.5)    
    with open("vfeat.yaml") as stream:
        cfg = yaml.safe_load(stream)
    model_sync = instantiate_from_config(cfg)
    ckpt = torch.load("pretrain/SyncFormer.pt", map_location="cpu")
    new_dict = {
        k.replace('module.v_encoder.', ''): v for k, v in ckpt["state_dict"].items() if 'v_encoder' in k
    }
    model_sync.load_state_dict(new_dict, strict=True)
    # Ensure the model is in evaluation mode
    model_sync.cuda().eval()
    
    process(args)
