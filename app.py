import sys
from share import *
import json
import yaml
import shutil
import cv2, os
import einops
import numpy as np
import torch
import argparse
import random
import json
import psutil
from datetime import datetime
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

from SyncFormer.utils import instantiate_from_config
from CodePredictor.generate_code import VideoCodeGenerator

from transformers import CLIPImageProcessor, CLIPModel

import gradio as gr
os.environ["GRADIO_TEMP_DIR"] = "./tmp"
N_PROMPT = ""

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='./models/cldm_v15.yaml')
parser.add_argument("--server-name", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", type=bool, default=False)

parser.add_argument("--save-path", default="vis/tmp")
parser.add_argument("--ckpt", type=str, default="pretrain/MelQCD.ckpt")

parser.add_argument("--num_quan", type=int, default=3, help="number of qmel values")
parser.add_argument("--num_win", type=int, default=8, help="number of frequency windows")
parser.add_argument("--num_vstar", type=int, default=32)

args = parser.parse_args()

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

def get_video_features(frames, model_sync):
    num_frames = len(frames)
    frames = torch.cat([frames[0].unsqueeze(0).repeat(8, 1, 1, 1), frames, frames[-1].unsqueeze(0).repeat(7, 1, 1, 1)], axis=0)
    segment_indices = (torch.arange(16).unsqueeze(0) + torch.arange(num_frames).unsqueeze(1)).cuda()
    segments = frames[segment_indices]
    segments = segments.permute(0, 2, 1, 3, 4).unsqueeze(0)
    rst, _ = model_sync(segments, for_loop=True)
    return rst[0]


class MelQCD:
    def __init__(self):
        # config dirs
        self.basedir = os.getcwd()
        self.model_dir = os.path.join(self.basedir, args.ckpt)
        self.savedir = os.path.join(self.basedir, args.save_path, datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        os.makedirs(self.savedir, exist_ok=True)
        self.loaded = False
        self.load_model()

    def load_model(self):
        # gr.Info("Start Load Models...")
        print("Start Load Models...")

        # load clip and syncformer model
        self.model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor_clip = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model_clip.cuda().eval()

        self.processor_sync = CLIPImageProcessor(image_mean=0.5, image_std=0.5)    
        with open("SyncFormer/vfeat.yaml") as stream:
            cfg = yaml.safe_load(stream)
        self.model_sync = instantiate_from_config(cfg)
        ckpt = torch.load("SyncFormer/pretrain/SyncFormer.pt", map_location="cpu")
        new_dict = {
            k.replace('module.v_encoder.', ''): v for k, v in ckpt["state_dict"].items() if 'v_encoder' in k
        }
        self.model_sync.load_state_dict(new_dict, strict=True)
        self.model_sync.cuda().eval()

        # load code predictor model
        self.vcgen = VideoCodeGenerator(args, 'CodePredictor/pretrain/CodePredictor.ckpt')

        self.melqcd_model = create_model_args(args.config, args.num_vstar, 768)
        self.melqcd_model.load_state_dict(load_state_dict(args.ckpt, location='cuda'), strict=True)
        self.melqcd_model = self.melqcd_model.cuda()
        
        self.vocoder = Generator.from_pretrained('.', subfolder="pretrain").cuda()

        # gr.Info("Load Finish!")
        print("Load Finish!")
        self.loaded = True

        return "Load"

    def foley(
        self,
        input_video,
        prompt_textbox,
        negative_prompt_textbox,
        sample_step_slider,
        cfg_scale_slider,
        seed_textbox,
    ):    

        # prepare features
        frames_pad = process_video(input_video)

        # clip
        frames_p = self.processor_clip(np.array(frames_pad), return_tensors='pt').to('cuda')
        with torch.no_grad():
            rst_clip = self.model_clip.get_image_features(**frames_p).cpu().numpy()
        print('extract clip feature down')
        
        # syncformer
        for i in range(256 - len(frames_pad)):
            frames_pad.append(np.zeros_like(frames_pad[-1]))
            
        frames_p = self.processor_sync(np.array(frames_pad), return_tensors='pt').to('cuda')
        with torch.no_grad():
            rst_sync = get_video_features(frames_p['pixel_values'], self.model_sync).cpu().numpy()
        print('extract syncformer feature down')

        # code predictor
        effect_length = min(rst_sync.shape[0], 250)
        effect_length = torch.tensor(effect_length)[None].cuda()

        rst_sync = torch.from_numpy(rst_sync)
        rst_sync = F.pad(rst_sync[:effect_length], (0, 0, 0, 250-effect_length), value=0)
        rst_sync = rst_sync[None].cuda()
        prediction, pred_mean, pred_std, tseq = self.vcgen.generate(rst_sync, effect_length)
        print('code prediction down')

        emb_pred = tseq[0]
        mean_pred = pred_mean[0]
        std_pred = pred_std[0]

        # melqcd
        ## create controlmap
        H, W, C = 256, 1024, 3
        upper_index = torch.linspace(0, 999, 1024).long()
        Mean = mean_pred[upper_index].unsqueeze(0).unsqueeze(-1).repeat(8, 1, 3)
        Std = std_pred[upper_index].unsqueeze(0).unsqueeze(-1).repeat(8, 1, 3)
        Spec = F.pad(emb_pred, (0, 0, 0, 1024-emb_pred.size(0)), 'constant', 0) # 1024, 8
        Spec = Spec.unsqueeze(-1).repeat(1, 1, 3).permute(1, 0, 2) # 8, 1024, 3  
        f_control = Mean + Std * Spec
        whole_map = torch.zeros(H, W, C)
        for env in range(8):
            whole_map[env*H//8: (env+1)*H//8] = f_control[env].unsqueeze(0).repeat(H//8, 1, 1)
        Control_map = whole_map 
        Control_map = torch.stack([Control_map for _ in range(1)], dim=0).float()
        Control_map = einops.rearrange(Control_map, 'b h w c -> b c h w').clone().cuda()
        Control_map = Control_map * 2 - 1
        print('create controlmap down')

        ## create Textual inversion feature
        rst_clip = torch.from_numpy(rst_clip)
        global_video_feat_mean = torch.mean(rst_clip, dim=0)[None]
        global_video_feat_mean = global_video_feat_mean.unsqueeze(0).cuda()
        print('create textual inversion feature down')

        ## start melqcd inference
        ddim_sampler = PLMSSampler(self.melqcd_model)
        with torch.no_grad():
            seed_everything(seed_textbox)
            cond = {
                "c_scale_mean": [global_video_feat_mean],
                "c_concat": [Control_map], 
                "c_crossattn": [[prompt_textbox]],
                }
            un_cond = {
                "c_scale_mean": [global_video_feat_mean],
                "c_concat": [Control_map], 
                "c_crossattn": [self.melqcd_model.get_learned_conditioning([negative_prompt_textbox] * 1)]}
            shape = (4, 256 // 8, 1024 // 8)

            strength = 1.
            self.melqcd_model.control_scales = ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(sample_step_slider, 1,
                                                        shape, cond, verbose=False, eta=0,
                                                        unconditional_guidance_scale=cfg_scale_slider,
                                                        unconditional_conditioning=un_cond)

            x_samples = self.melqcd_model.decode_first_stage(samples)[0]
        print('melqcd inference down')

        mel_path = os.path.join(self.savedir, 'mel')
        wav_path = os.path.join(self.savedir, 'audio')
        video_path = os.path.join(self.savedir, 'video')

        os.makedirs(mel_path, exist_ok=True)
        os.makedirs(wav_path, exist_ok=True)
        os.makedirs(video_path, exist_ok=True)
        name = "output"

        results = ((x_samples+1) / 2).clip(0, 1)
        audio_lst = results.squeeze().cpu().numpy()
        np.save(mel_path + f'/{name}.npy', audio_lst) 
        res_mel = denormalize_spectrogram(results)
        audio = self.vocoder.inference(res_mel, lengths=160000)[0]
        audio = audio[:int(10 * 16000)] 
        sf.write(wav_path + f'/{name}.wav', audio, 16000)
        video_clip = VideoFileClip(input_video)
        audio_clip = AudioFileClip(wav_path + f'/{name}.wav')
        audio_clip = audio_clip.subclip(0, 10)
        video_with_new_audio = video_clip.set_audio(audio_clip)
        video_with_new_audio.write_videofile(video_path + f'/{name}.mp4', codec='libx264', audio_codec='aac')
        print('save results down')

        return video_path + f'/{name}.mp4'


controller = MelQCD()
with gr.Blocks(css=css) as demo:
    gr.HTML(
        '<h1 style="height: 136px; display: flex; align-items: center; justify-content: center;"><strong style="font-size: 30px;">Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition</strong></h1>'
    )
    gr.HTML(
        '<p id="authors" style="text-align:center; font-size:24px;"> \
        <a href="">Juncheng Wang</a><sup>1 *</sup>,&nbsp \
        <a href="">Chao Xu</a><sup>2 *</sup>,&nbsp \
        <a href="">Cheng Yu</a><sup>2 *</sup>,&nbsp \
        <a href="">Lei Shang</a><sup>2 †</sup>,&nbsp \
        <a href="">Zhe Hu</a><sup>1</sup>,&nbsp \
        <a href="">Shujun Wang</a><sup>1 ‡</sup>,&nbsp \
        <a href="">Liefeng Bo</a><sup>2 ‡</sup>\
        <br>\
        <span>\
            <sup>1</sup>Hong Kong Polytechnic University &nbsp;&nbsp;&nbsp;\
            <sup>2</sup>Tongyi Lab, Alibaba Group &nbsp;&nbsp;&nbsp;\
        </span>\
        <br>\
        <span>\
            *Equal contribution, †Project lead, ‡Corresponding author\
        </span>\
    </p>'
    )
    with gr.Row():
        gr.Markdown(
            "<div align='center'><font size='5'><a href=''>Project Page</a> &ensp;"  # noqa
            "<a href=''>Paper</a> &ensp;"
            "<a href=''>Code</a> &ensp;"
            "<a href=''>Demo</a> </font></div>"
        )

    with gr.Column(variant="panel"):
        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Row():
                    init_img = gr.Video(label="Input Video")
                with gr.Row():
                    prompt_textbox = gr.Textbox(value="", label="Prompt", lines=1)
                with gr.Row():
                    negative_prompt_textbox = gr.Textbox(value=N_PROMPT, label="Negative prompt", lines=1)

                with gr.Accordion("Sampling Settings", open=False):
                    with gr.Row():
                        
                        sample_step_slider = gr.Slider(
                            label="Sampling steps", value=100, minimum=10, maximum=100, step=1
                        )
                    cfg_scale_slider = gr.Slider(label="CFG Scale", value=7.5, minimum=0, maximum=20)

                with gr.Row():
                    seed_textbox = gr.Textbox(label="Seed", value=42)
                    seed_button = gr.Button(value="\U0001f3b2", elem_classes="toolbutton")
                seed_button.click(fn=lambda x: random.randint(1, 1e8), outputs=[seed_textbox], queue=False)

                generate_button = gr.Button(value="Generate", variant="primary")

            with gr.Column():
                result_video = gr.Video(label="Generated Audio", interactive=False)
                with gr.Row():
                    gr.Markdown(
                        "<div style='word-spacing: 6px;'><font size='5'><b>Notes</b>: <br> \
                        1. The model used in this demo is trained on VGGSound by default. For more details, please refer to the github page.<br>\
                    "
                    )

        generate_button.click(
            fn=controller.foley,
            inputs=[
                init_img,
                prompt_textbox,
                negative_prompt_textbox,
                sample_step_slider,
                cfg_scale_slider,
                seed_textbox,
            ],
            outputs=[result_video],
        )
        '''
        gr.Examples(
            examples=[
                ["examples/video_clip/184825_0.mp4", "francolin calling", "", 100, 7.5, 42],
                # ["examples/video_clip/185302_0.mp4", "volcano explosion", "", 100, 7.5, 42],
                # ["examples/video_clip/191517_0.mp4", "planing timber", "", 100, 7.5, 42],
                # ["examples/video_clip/194948_0.mp4", "train horning", "", 100, 7.5, 42],
            ],
            inputs=[
                init_img,
                prompt_textbox,
                negative_prompt_textbox,
                sample_step_slider,
                cfg_scale_slider,
                seed_textbox,
            ],
            cache_examples=True,
            outputs=[result_video],
            fn=controller.foley,
        )
        '''

    demo.queue(10)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
      