import os
import sys
import json
import re
import random
import yaml
import shutil
import numpy as np
import torch
import torch.utils.data
import torchaudio
import librosa
import argparse
import cv2 as cv
from glob import glob
from tqdm import tqdm

from scipy.io import wavfile
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
from scipy.io import wavfile
from scipy.io.wavfile import read
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
from torch.nn import functional as F
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, AudioFileClip

def read_wav(wav_path):
    audio_clip = AudioFileClip(wav_path)
    audio_sample_rate = audio_clip.fps
    audio_array = audio_clip.to_soundarray()
    if len(audio_array.shape) == 2 and audio_array.shape[1] != 1:
            # If the audio has multiple channels, convert it to mono
        audio_array = np.mean(audio_array, axis=1) # (133182, )
    audio_clip.close()
    return audio_array, audio_sample_rate

def read_mp4(mp4_path):
    video_clip = VideoFileClip(mp4_path)
    frames = np.array([frame for frame in video_clip.iter_frames()])
    # Print the audio sample rate
    # Get the duration of the video in seconds
    video_duration = video_clip.duration
    # Print the video duration
    # Optionally, you can convert the audio to a sound array
    # Close the clips to release resources
    video_clip.close()
    return frames, video_duration, video_clip.fps

def resample_audio(audio_array, temp_sample_rate):
    if temp_sample_rate != TARGET_SR:
        audio_array = torchaudio.functional.resample(audio_array, orig_freq=temp_sample_rate, new_freq=TARGET_SR)
    return audio_array

def mel_spectrogram_HiFiGAN(audio, num_mels, sampling_rate, target_mel_length, n_fft=1024, hop_size=160, win_size=1024, fmin=0, fmax=8e3, center=False):
    audio = audio / MAX_WAV_VALUE
    max_val = torch.max(torch.abs(audio))
    audio = audio / max_val * 0.95


    mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
    mel_basis = torch.from_numpy(mel).float().to(audio.device)
    audio = torch.nn.functional.pad(audio.unsqueeze(0), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect').squeeze(1)
    spec = torch.stft(audio, n_fft, hop_length=hop_size, win_length=win_size, window=torch.hann_window(win_size).to(audio.device),
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(spec, min=1e-5))
    if mel_spec.shape[2] < target_mel_length:
        mel_spec = torch.nn.functional.pad(mel_spec, (0, target_mel_length - mel_spec.shape[2]), "constant", 0)
        
    return mel_spec[..., :target_mel_length]

def forward(mp4_path, wav_path):
    audio_array, audio_sample_rate = read_wav(wav_path)
    audio = torch.from_numpy(audio_array).float().cuda()

    audio = resample_audio(audio, audio_sample_rate)
    mel = mel_spectrogram_HiFiGAN(audio, FREQ_BIN, TARGET_SR, MEL_LENGTH).cpu().numpy()
    
    return {'mels': mel}

def args_parse():
    args = argparse.ArgumentParser()
    args.add_argument("--root", type=str, default='', help="root path")
    args = args.parse_args()
    return args

FREQ_BIN = 256
TARGET_SR = int(1.6e4)
TARGET_DURATION = 10
MAX_WAV_VALUE = 32768.0
MEL_LENGTH = 1024
TARGET_FPS = 25
BIN_Thre = 0.05
if __name__ == '__main__':
    args = args_parse()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()[None, :, None, None] 
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()[None, :, None, None] 

    tar_root = args.root
    video_data_root = os.path.join(tar_root, 'video_clip')
    audio_data_root = os.path.join(tar_root, 'audio_clip')
    os.makedirs(audio_data_root, exist_ok=True)

    video_paths = sorted(glob(video_data_root + '/*.mp4'))

    mel_path = os.path.join(tar_root, 'mels')
    os.makedirs(mel_path, exist_ok=True)

    for i, video_path in tqdm(enumerate(video_paths), desc=f'{len(video_paths)}'):

        base_name = os.path.basename(video_path)[:-4]
        audio_path = os.path.join(audio_data_root, base_name+'.wav')
        if not os.path.exists(audio_path):
            audio_command = 'ffmpeg -i \"{}\" -loglevel error -y -f wav -acodec pcm_s16le ' \
                        '-ar 16000 \"{}\"'.format(video_path, audio_path)
            os.system(audio_command)

        res = forward(video_path, audio_path)
        assert res['mels'].shape == (1, FREQ_BIN, MEL_LENGTH), res['mels'].shape
        np.save(os.path.join(mel_path, f"{base_name}.npy"), res['mels'])