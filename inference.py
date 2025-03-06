from share import *
import json
import shutil
import cv2, os
import einops
import numpy as np
import torch
import argparse
import random
import argparse
import json
import psutil
import soundfile as sf
from os import path as osp
from copy import deepcopy
from einops import rearrange, repeat
from torch.nn import functional as F
from cldm.ddim_hacked import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from pytorch_lightning import seed_everything
from cldm.model import create_model_args, load_state_dict
from moviepy.editor import VideoFileClip, AudioFileClip
from foleycrafter.pipelines.auffusion_pipeline import Generator

def normalize_spectrogram(
    spectrogram: torch.Tensor,
    max_value: float = 200,
    min_value: float = 1e-5,
    power: float = 1.0,
) -> torch.Tensor:
    # Rescale to 0-1
    max_value = np.log(max_value)  # 5.298317366548036
    min_value = np.log(min_value)  # -11.512925464970229
    spectrogram = torch.clamp(spectrogram, min=min_value, max=max_value)
    data = (spectrogram - min_value) / (max_value - min_value)
    # Apply the power curve
    data = torch.pow(data, power)
    # 1D -> 3D
    data = data.repeat(3, 1, 1)
    # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
    data = torch.flip(data, [1])

    return data, min_value, max_value



def denormalize_spectrogram(
    data: torch.Tensor,
    max_value: float = 200, 
    min_value: float = 1e-5, 
    power: float = 1):
    
    assert len(data.shape) == 3, "Expected 3 dimensions, got {}".format(len(data.shape))

    max_value = np.log(max_value)
    min_value = np.log(min_value)
    data = torch.flip(data, [1])    
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)        
    assert data.shape[0] == 3, "Expected 3 channels, got {}".format(data.shape[0])
    data = data[0]
    data = torch.pow(data, 1 / power)
    spectrogram = data * (max_value - min_value) + min_value

    return spectrogram

def prepare_control_signal(sample_name, num_samples=None):
    sample_name = str(sample_name)
    H, W, C = 256, 1024, 3

    upper_index = torch.linspace(0, 999, 1024).long()
    Mean = torch.tensor(np.load(
        osp.join(set_root, 'rst_mean', f'{sample_name}.npy'),  #'rst_mean' 'PredictwoSwap', 'Mean'
    ))
    Mean = Mean[upper_index].unsqueeze(0).unsqueeze(-1).repeat(8, 1, 3)
    Std = torch.tensor(np.load(
        osp.join(set_root, 'rst_std', f'{sample_name}.npy'), # 'rst_std' 'PredictwoSwap', 'Std'
    ))
    Std = Std[upper_index].unsqueeze(0).unsqueeze(-1).repeat(8, 1, 3)
    Spec = torch.tensor(np.load(
        osp.join(set_root, 'rst_emb', f'{sample_name}.npy'), # 'rst_emb' 'PredictwoSwap', 'Class38' 
    ))
    Spec = F.pad(Spec, (0, 0, 0, 1024-Spec.size(0)), 'constant', 0) # 1024, 8
    Spec = Spec.unsqueeze(-1).repeat(1, 1, 3).permute(1, 0, 2) # 8, 1024, 3  
    f_control = Mean + Std * Spec
    whole_map = torch.zeros(H, W, C)
    for env in range(8):
        whole_map[env*H//8: (env+1)*H//8] = f_control[env].unsqueeze(0).repeat(H//8, 1, 1)
    Control_map = whole_map        

    assert Control_map.shape == (256, 1024, 3), Control_map.shape
       
    Control_map = torch.stack([Control_map for _ in range(num_samples)], dim=0).float()
    Control_map = einops.rearrange(Control_map, 'b h w c -> b c h w').clone().cuda()
    return (Control_map * 2 - 1)

def process(set_root=None, sample_id=None, prompt=None, 
    a_prompt=None, n_prompt=None, num_samples=None,
    ddim_steps=100, guess_mode=False, 
    strength=1., scale=7.5, seed=-1, eta=0):
    sample_name = str(sample_id)
    
    negative_prompt = ''
    prompt = data_dict[sample_name].replace('_', ' ')
    positive_prompt = ''
    
    global_video_feat = torch.tensor(np.load(os.path.join(set_root, 'CLIP_LargeEmbed', f'{sample_name}.npy')))
    global_video_feat_mean = torch.mean(global_video_feat, dim=0)[None]
    global_video_feat_mean = global_video_feat_mean.unsqueeze(0).cuda()

    Control_map = prepare_control_signal(sample_name, num_samples)

    with torch.no_grad():
        seed_everything(args.seed)
        cond = {
            "c_scale_mean": [global_video_feat_mean],
            "c_concat": [Control_map], 
            "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)] if woTI else [[prompt]],
            }
        un_cond = {
            "c_scale_mean": None if guess_mode else [global_video_feat_mean],
            "c_concat": None if guess_mode else [Control_map], 
            "c_crossattn": [model.get_learned_conditioning([negative_prompt] * num_samples)]}
        shape = (4, 256 // 8, 1024 // 8)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples) # b c h w
        return x_samples[0]

def Spec2Audio(results, wav_path, video_path, output_mel):
    results = ((results+1) / 2).clip(0, 1)
    audio_lst = results.squeeze().cpu().numpy()
    np.save(output_mel, audio_lst) 
    res_mel = denormalize_spectrogram(results)
    audio = vocoder.inference(res_mel, lengths=160000)[0]
    audio = audio[:int(10 * 16000)] 
    sf.write(wav_path, audio, 16000)
    video_clip = VideoFileClip(osp.join(video_root, f'{sample_idx}.mp4'))
    audio_clip = AudioFileClip(wav_path)
    audio_clip = audio_clip.subclip(0, 10)
    video_with_new_audio = video_clip.set_audio(audio_clip)
    video_with_new_audio.write_videofile(video_path, codec='libx264', audio_codec='aac')

   
def parse_args():
    parser = argparse.ArgumentParser(
        description="DDP Inference to Sample",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=str, default='')
    parser.add_argument("--num_vstar", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--woTI", type=bool, default=False)
    parser.add_argument("--model_ckpt", type=str, default="pretrain/MelQCD.ckpt")
    parser.add_argument("--output_dir", type=str, default="vis/tmp")
    parser.add_argument("--prompt_dir", type=str, default=None)
    parser.add_argument("--DeviceIdx", type=int, default=0)
    
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
       
    args = parse_args()
    p = psutil.Process()
    p.cpu_affinity(list(np.arange(args.DeviceIdx*12, (args.DeviceIdx+1)*12)))
    model = create_model_args('./models/cldm_v15.yaml', args.num_vstar, 768).cpu()
    model.load_state_dict(load_state_dict(args.model_ckpt, location='cuda'), strict=True)
    model = model.cuda()
    # model = model.eval()
    ddim_sampler = PLMSSampler(model)
    vocoder      = Generator.from_pretrained('.', subfolder="pretrain").cuda()
    
    woTI = args.woTI
    set_root = args.root
    video_root = args.root + "/video_clip"
    
    sample_names = [name[:-4] for name in sorted(os.listdir(video_root))]
    if args.prompt_dir is not None:
        with open(args.prompt_dir, 'r') as f:
            data_dict = json.load(f)
    else:
        data_dict = {}
        for name in sample_names:
            data_dict[name] = ''
        print(data_dict)

    output_root = args.output_dir
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(output_root+'/video', exist_ok=True)
    os.makedirs(output_root+'/audio', exist_ok=True)
    os.makedirs(output_root+'/mel', exist_ok=True)
    
    for num_index, sample_idx in enumerate(sample_names):
        print(f"Processing [{num_index}|{len(sample_names)}]...")
        x_samples = process(set_root, sample_idx, num_samples=1)
        wav_path, video_path = osp.join(output_root, 'audio', f'{sample_idx}.wav'), osp.join(output_root, 'video', f'{sample_idx}_{data_dict[sample_idx]}.mp4')
    
        mel_path = osp.join(output_root, 'mel', f'{sample_idx}.npy')
        Spec2Audio(x_samples, wav_path, video_path, mel_path)
      