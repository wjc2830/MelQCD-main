import numpy as np
from tqdm import tqdm
import os
from einops import rearrange
import sys
import json
import argparse

def normalize_spectrogram(spectrogram, 
    max_value=200, min_value=1e-5, power=1.0):
    # H, W
    max_value = np.log(max_value)  # 5.298317366548036
    min_value = np.log(min_value)  # -11.512925464970229
    if spectrogram.shape[1] != 1024:
        spectrogram = np.pad(spectrogram, 
            ((0, 0), (0, 1024 - spectrogram.shape[1])), 'constant', constant_values=min_value)

    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=max_value)
    # spectrogram = spectrogram[4:256:8]
    data = (spectrogram - min_value) / (max_value - min_value)
    data = np.flip(data, axis=0)
    return data

def quantize_spectrogram(spectrogram, clamp_value=3):
    # H, W
    Mean, Std = np.mean(spectrogram, axis=0, keepdims=True), \
        np.std(spectrogram, axis=0, keepdims=True)
    Std = np.where(Std==0, np.ones_like(Std), Std)
    spec = np.around((spectrogram - Mean) / Std)
    Min, Max = spec.min(), spec.max()
    spec = np.clip(spec, a_min=-clamp_value, a_max=clamp_value)
    return spec, Mean, Std, Min, Max

def norm_and_quantize(mel, clamp_val=3):
    normed_mel = normalize_spectrogram(mel)
    spec_class, spec_m, spec_s, Min, Max = \
        quantize_spectrogram(normed_mel, clamp_val)
    return spec_class, spec_m, spec_s, Min, Max


def args_parse():
    args = argparse.ArgumentParser()
    args.add_argument("--root", type=str, default='', help="root path")
    args = args.parse_args()
    return args


NumQuan = 3
NumWin = 8
assert NumQuan % 2 == 1
assert NumWin % 4 == 0
NumSide = (NumQuan - 1)//2

if __name__ == '__main__':
    args = args_parse()
    save_dir = args.root
    file_names = sorted(os.listdir(os.path.join(save_dir, 'video_clip')))
        
    mel_root = os.path.join(save_dir, 'mels',)
    sub_dirs = ['Mean', 'Std', 'Emb', 'Code']
    save_root = os.path.join(save_dir, 'Qmel_{}_{}'.format(NumQuan, NumWin))
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(save_root, sub_dir), exist_ok=True)
            
    for save_name in tqdm(file_names):
        save_name = save_name[:-4]
        try:
            mel = np.load(os.path.join(mel_root, save_name + '.npy')).squeeze()
            H, W = mel.shape
            # print(mel.shape)
            assert H == 256
            assert W <= 1024
            mel = rearrange(mel, '(a b) w -> a b w', a=NumWin, b=H//NumWin).mean(axis=1)
            spec_class, spec_m, spec_s, Min, Max = norm_and_quantize(mel, NumSide)

            np.save(os.path.join(save_root, 'Mean', '{}.npy'.format(save_name)), spec_m[0])
            np.save(os.path.join(save_root, 'Std', '{}.npy'.format(save_name)), spec_s[0])
            # W * NumWin
            np.save(os.path.join(save_root, 'Emb', '{}.npy'.format(save_name)), spec_class.transpose(1, 0))
            codes = rearrange(spec_class.transpose(1, 0), 'w (a b) -> w a b', a=NumWin//4, b=4)
            mult = np.power(NumQuan, np.arange(4))
            codes = np.sum((codes + NumSide) * mult[None, None, :], axis=2)      
            np.save(os.path.join(save_root, 'Code', '{}.npy'.format(save_name)), codes.astype(int))
        except Exception as e:
            print(e)

