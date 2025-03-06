import os
import cv2 as cv
import numpy as np
import torch
import torchvision
from PIL import Image
from evaluation.onset import detect_onset
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, save_dir='VideoRes'):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.save_dir = save_dir
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.cached_steps = []
        
    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", self.save_dir)
        all_images = None
        os.makedirs(root, exist_ok=True)
        tmp_path = os.path.join(root, "gs-{:06}b-{:06}.png".format(global_step, batch_idx))
        for num_k, k in enumerate(images):
            if k in ['reconstruction', 'samples_cfg_scale_9.00']:
                images[k] = torch.cat([denormalize_spectrogram(images[k][i]).unsqueeze(0).unsqueeze(0) for i in range(images[k].size(0))], dim=0)
            elif k == "control":
                images[k] = torch.zeros_like(images['reconstruction'])
            else:
                images[k] = (images[k] * 255)
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy().astype(np.uint8)
            
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            
            if all_images is None:
                H, W, _ = grid.shape
                all_images = np.zeros((4*H, W, 3))
            all_images[num_k*H: (num_k+1)*H] = grid
        Image.fromarray(np.uint8(all_images)).save(tmp_path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = pl_module.current_epoch
        if (self.check_frequency(check_idx) and 
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                check_idx not in self.cached_steps and
                self.max_images > 0):
            logger = type(pl_module.logger)
            self.cached_steps.append(check_idx)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0 and check_idx != 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")


def denormalize_spectrogram(
        data: torch.Tensor,
        max_value: float = 200, 
        min_value: float = 1e-5, 
        power: float = 1, 
    ) -> torch.Tensor:
        
        assert len(data.shape) == 3, "Expected 3 dimensions, got {}".format(len(data.shape))

        max_value = np.log(max_value)
        min_value = np.log(min_value)
        # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
        data = torch.flip(data, [1])    
        if data.shape[0] == 1:
            data = data.repeat(3, 1, 1)        
        assert data.shape[0] == 3, "Expected 3 channels, got {}".format(data.shape[0])
        data = data[0]
        # Reverse the power curve
        data = torch.pow(data, 1 / power)
        # Rescale to max value
        spectrogram = data * (max_value - min_value) + min_value

        return spectrogram