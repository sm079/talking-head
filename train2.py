# MODULES
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)

class PixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
    
class DepthToSpace(nn.Module):
    def __init__(self, size):
        super().__init__()
        
        self.size = size

    def forward(self, x):
        b, c, h, w = x.shape
        oh, ow = h * self.size, w * self.size
        oc = c // (self.size * self.size)
        x = x.view(b, self.size, self.size, oc, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2)
        x = x.contiguous().view(b, oc, oh, ow)
        return x
    
class Res(nn.Module):
    def __init__(self, n_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1),
        )

        self.fuse = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.fuse(x + self.conv(x))
    
class OutConv(nn.Module):
    def __init__(self, n_ch):
        super().__init__()

        self.out_conv = nn.ModuleList([
            nn.Conv2d(n_ch, 3, kernel_size=1),
            nn.Conv2d(n_ch, 3, kernel_size=3, padding=1),
            nn.Conv2d(n_ch, 3, kernel_size=3, padding=1),
            nn.Conv2d(n_ch, 3, kernel_size=3, padding=1),
        ])

    def forward(self, x):
        return torch.cat([i(x) for i in self.out_conv], dim=1)

class Encoder(nn.Module):
    def __init__(self, e_ch):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(     3, e_ch*1, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.1, inplace=True), Res(e_ch*1),
            nn.Conv2d(e_ch*1, e_ch*2, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(e_ch*2, e_ch*4, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(e_ch*4, e_ch*8, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(e_ch*8, e_ch*8, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.1, inplace=True), Res(e_ch*8),
            nn.Flatten(), PixelNorm()
        )

    def forward(self, x):
        return self.image_encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, ae_dim, d_ch, m_ch):
        super().__init__()

        self.image_decoder = nn.Sequential(
            nn.Conv2d(ae_dim, d_ch*8*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2), Res(d_ch*8),
            nn.Conv2d(d_ch*8, d_ch*8*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2), Res(d_ch*8),
            nn.Conv2d(d_ch*8, d_ch*4*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2), Res(d_ch*4),
            nn.Conv2d(d_ch*4, d_ch*2*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2), Res(d_ch*2),
            OutConv(d_ch*2), DepthToSpace(2), nn.Sigmoid()
        )

        self.mask_decoder = nn.Sequential(
            nn.Conv2d(ae_dim, m_ch*8*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2),
            nn.Conv2d(m_ch*8, m_ch*8*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2),
            nn.Conv2d(m_ch*8, m_ch*4*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2),
            nn.Conv2d(m_ch*4, m_ch*2*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2),
            nn.Conv2d(m_ch*2, m_ch*1*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2),
            nn.Conv2d(m_ch*1, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, z):
        return self.image_decoder(z), self.mask_decoder(z)
    
class UNetPatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=16):
        super().__init__()
        
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(base_ch *1, base_ch* 2, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(base_ch *2, base_ch* 4, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(base_ch *4, base_ch* 8, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(base_ch *8, base_ch*16, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(base_ch*16, base_ch*32, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)),
        ])
        
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(base_ch*32, base_ch*16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.ConvTranspose2d(base_ch*32, base_ch* 8, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.ConvTranspose2d(base_ch*16, base_ch* 4, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.ConvTranspose2d(base_ch* 8, base_ch* 2, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.ConvTranspose2d(base_ch* 4, base_ch* 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2, inplace=True))
        ])

        self.in_conv     = nn.Conv2d(in_ch     ,    base_ch, kernel_size=1, padding=0)
        self.out_conv    = nn.Conv2d(base_ch* 2,          1, kernel_size=1, padding=0)
        self.center_out  = nn.Conv2d(base_ch*32,          1, kernel_size=1, padding=0)
        self.center_conv = nn.Conv2d(base_ch*32, base_ch*32, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = nn.functional.leaky_relu(self.in_conv(x), 0.2)
        
        # Encoder
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)
        
        # Center
        center_out = self.center_out(x)
        x = nn.functional.leaky_relu(self.center_conv(x), 0.2)
        
        # Decoder
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            if i < len(self.decoder) - 1:  # Skip connection for all but the last layer
                x = torch.cat([encoder_outputs[-i-2], x], dim=1)
        
        x = torch.cat([x, x], dim=1)  # Concatenate with itself to double channels
        x = self.out_conv(x)
        
        return center_out, x

# LOSSES + OPTIMIZERS
import math
import lpips
from pytorch_msssim import MS_SSIM
from torch.optim import Optimizer
import numpy as np

class MSSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 1 - super(MSSIM_Loss, self).forward(img1, img2)

class AdaBelief(Optimizer):
    def __init__(self, params, lr=0.001, beta_1=0.9, beta_2=0.999, lr_dropout=1.0, lr_cos=0, clipnorm=0.0, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("Invalid beta_1 parameter: {}".format(beta_1))
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("Invalid beta_2 parameter: {}".format(beta_2))
        if not 0.0 <= lr_dropout <= 1.0:
            raise ValueError("Invalid lr_dropout parameter: {}".format(lr_dropout))
        if not 0.0 <= clipnorm:
            raise ValueError("Invalid clipnorm value: {}".format(clipnorm))
        if not 0.0 <= lr_cos:
            raise ValueError("Invalid lr_cos value: {}".format(lr_cos))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, eps=eps)
        super(AdaBelief, self).__init__(params, defaults)
        self.iterations = 0

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.iterations += 1

        for group in self.param_groups:
            lr = group['lr']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            lr_dropout = group['lr_dropout']
            lr_cos = group['lr_cos']
            clipnorm = group['clipnorm']
            eps = group['eps']

            if lr_cos != 0:
                lr *= (math.cos(self.iterations * (2 * math.pi / float(lr_cos))) + 1.0) / 2.0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if lr_dropout != 1.0:
                        state['lr_rnd'] = torch.bernoulli(torch.full_like(p.data, lr_dropout))

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Apply gradient clipping if needed
                if clipnorm > 0.0:
                    grad = torch.nn.utils.clip_grad_norm_(p, clipnorm)

                # Update biased first moment estimate
                exp_avg.mul_(beta_1).add_(grad, alpha=1 - beta_1)

                # Update second moment estimate
                grad_diff = grad - exp_avg
                exp_avg_sq.mul_(beta_2).addcmul_(grad_diff, grad_diff, value=1 - beta_2)

                # Compute bias-corrected estimates
                bias_correction1 = 1 - beta_1 ** state['step']
                bias_correction2 = 1 - beta_2 ** state['step']

                adapted_lr = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = (exp_avg_sq.sqrt() + eps)
                step_size = adapted_lr

                step = exp_avg / denom
                if lr_dropout != 1.0:
                    step = step * state['lr_rnd']

                p.data.add_(-step_size, step)

        return loss

# DATA LOADER
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader


class SrcDstDataset(IterableDataset):
    def __init__(
            self, 
            src_root_dirs,
            dst_root_dirs,
            image_size, 
            src_rotation_range, 
            dst_rotation_range, 
            src_scale_range, 
            dst_scale_range, 
            src_random_flip,
            dst_random_flip,
            src_random_hue,
            dst_random_hue,
            ):
        
        self.image_size = image_size

        self.src_rotation_range = src_rotation_range
        self.dst_rotation_range = dst_rotation_range
        self.src_scale_range = src_scale_range
        self.dst_scale_range = dst_scale_range 
        self.src_random_flip = src_random_flip
        self.dst_random_flip = dst_random_flip
        self.src_random_hue = src_random_hue
        self.dst_random_hue = dst_random_hue

        self.src_images, self.src_masks = [], []
        self.dst_images, self.dst_masks = [], []

        for root_dir in src_root_dirs:
            self.src_images += list((Path(root_dir)/"aligned").iterdir())
            self.src_masks  += list((Path(root_dir)/"masks").iterdir())

        for root_dir in dst_root_dirs:
            self.dst_images += list((Path(root_dir)/"aligned").iterdir())
            self.dst_masks  += list((Path(root_dir)/"masks").iterdir())

    def transform(self, image, mask, rotation_range, scale_range, random_flip, random_hue):
        # Random Rotate
        angle = random.randint(rotation_range[0], rotation_range[1])
        image = TF.rotate(img=image, angle=angle, interpolation=transforms.InterpolationMode.BILINEAR)
        mask  = TF.rotate(img=mask , angle=angle, interpolation=transforms.InterpolationMode.BILINEAR)

        # Center Crop
        crop_size = 448 + random.randint(-32, 32)
        image = TF.center_crop(image, (crop_size, crop_size))
        mask  = TF.center_crop(mask , (crop_size, crop_size))

        # Random Crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(img=image, scale=scale_range, ratio=[1.,1.])
        image = TF.resized_crop(
            image, i, j, h, w, size=[self.image_size,self.image_size], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        mask  = TF.resized_crop(
            mask , i, j, h, w, size=[self.image_size,self.image_size], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)

        # Random Flip
        if random_flip:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
        
        if random_hue > 0:
            hue_factor = random.uniform(-random_hue, random_hue)
            image = TF.adjust_hue(image, hue_factor)

        image = TF.to_tensor(image)
        mask  = TF.to_tensor(mask)
        return image, mask

    def __iter__(self):
        while True:
            src_idx = random.randint(0, len(self.src_images) - 1)
            dst_idx = random.randint(0, len(self.dst_images) - 1)

            src_image = Image.open(self.src_images[src_idx])
            src_mask  = Image.open(self.src_masks[src_idx]).convert("L")
            
            dst_image = Image.open(self.dst_images[dst_idx])
            dst_mask  = Image.open(self.dst_masks[dst_idx]).convert("L")

            src_image, src_mask = self.transform(
                image=src_image, 
                mask=src_mask, 
                rotation_range=self.src_rotation_range, 
                scale_range=self.src_scale_range,
                random_flip=self.src_random_flip,
                random_hue=self.src_random_hue
            )

            dst_image, dst_mask = self.transform(
                image=dst_image, 
                mask=dst_mask, 
                rotation_range=self.dst_rotation_range, 
                scale_range=self.dst_scale_range,
                random_flip=self.dst_random_flip,
                random_hue=self.dst_random_hue
            )

            yield src_image, src_mask, dst_image, dst_mask


def randomWarp(image_batch, strength: float = 0.2, num_divs: int = 5):
    if strength < 0.01:
        return image_batch
    
    image_size = image_batch.shape[-2:]
    grid_base  = torch.linspace(-1., 1., num_divs).broadcast_to((image_batch.shape[0],1,num_divs,num_divs))

    grid = torch.stack((
        F.interpolate(
            input = (grid_base + (strength/num_divs) * torch.randn(grid_base.shape)), 
            size  = image_size, mode="bilinear", align_corners=True).squeeze(),
        F.interpolate(
            input = grid_base.transpose(2,3) + (strength/num_divs) * torch.randn(grid_base.shape), 
            size  = image_size, mode="bilinear", align_corners=True).squeeze()
    ), dim=3)

    return F.grid_sample(image_batch, grid, mode='bilinear', padding_mode='border', align_corners=True)

# PREVIEW UTILS
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def save_input_preview(src_preview, src_preview_warped, src_preview_masked, dst_preview, dst_preview_warped, dst_preview_masked, dir):
    image_height, image_width, _ = src_preview[0].shape

    grid_height = len(src_preview) * image_height
    grid_width  = 6 * image_width
    grid_image  = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for i in range(len(src_preview)):
        grid_image[i*image_height:(i+1)*image_height, 0*image_width:1*image_width] = src_preview[i]
        grid_image[i*image_height:(i+1)*image_height, 1*image_width:2*image_width] = src_preview_warped[i]
        grid_image[i*image_height:(i+1)*image_height, 2*image_width:3*image_width] = src_preview_masked[i]
        grid_image[i*image_height:(i+1)*image_height, 3*image_width:4*image_width] = dst_preview[i]
        grid_image[i*image_height:(i+1)*image_height, 4*image_width:5*image_width] = dst_preview_warped[i]
        grid_image[i*image_height:(i+1)*image_height, 5*image_width:6*image_width] = dst_preview_masked[i]

    Image.fromarray(grid_image).save(f"{dir}/preview/in.jpg")

def save_output_preview(src_src, dst_dst, src_dst, it: int, dir):
    image_height, image_width, _ = src_src[0].shape

    grid_height = len(src_src) * image_height
    grid_width  = 3 * image_width
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for i in range(len(src_src)):
        grid_image[i*image_height:(i+1)*image_height, 0*image_width:1*image_width] = src_src[i]
        grid_image[i*image_height:(i+1)*image_height, 1*image_width:2*image_width] = dst_dst[i]
        grid_image[i*image_height:(i+1)*image_height, 2*image_width:3*image_width] = src_dst[i]

    Image.fromarray(grid_image).save(f"{dir}/preview/out_{it:06d}.jpg")
    Image.fromarray(grid_image).save(f"{dir}/preview/last.jpg")

def save_loss_plot(losses, dir, window_size=50, it=0):
    steps = losses["steps"]
    loss_names = list(losses.keys())
    loss_names.remove("steps")
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    plt.figure(figsize=(10, 6))
    for i, loss_name in enumerate(loss_names):
        color_index = i % len(colors)
        color = colors[color_index]
        label = loss_name.capitalize() + " Loss"
        if len(losses[loss_name]) > window_size:
            smoothed_loss = np.convolve(losses[loss_name], np.ones(window_size)/window_size, mode="valid")
            plt.plot(steps[:len(smoothed_loss)], smoothed_loss, color=color, label=label)
            label = "_nolegend_"
        plt.plot(steps, losses[loss_name], color=color, alpha=0.3, label=label)

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Losses Over Steps')
    plt.legend()
    plt.ylim(bottom=0, top=1.5*np.median(losses["dst"]))
    plt.grid(True)
    plt.savefig(f"{dir}/preview/loss_{it:06d}.jpg")
    plt.savefig(f"{dir}/preview/last_loss.jpg")
    plt.close()
    
# OTHER UTILS
def to_cpu(images):
    return (images * 255).permute(0, 2, 3, 1).to("cpu").to(torch.uint8).numpy()

def get_smooth_noisy_labels(label, tensor, smoothing=0.1, noise=0.):
    probs = torch.tensor([[1-noise, noise]]) if label == 0 else torch.tensor([[noise, 1-noise]])
    return (1 - smoothing) * torch.multinomial(probs.repeat(tensor.numel(), 1), 1).float().view_as(tensor)

def DLoss(labels, logits):
    return F.binary_cross_entropy_with_logits(logits, labels).mean()

def total_variation_mse(images):
    return (
        torch.sum((images[:, 1:, :, :] - images[:, :-1, :, :]) ** 2) + 
        torch.sum((images[:, :, 1:, :] - images[:, :, :-1, :]) ** 2)
    )

# CONFIGS
import yaml
import argparse
import warnings

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def save_config(config, filename):
    with open(filename, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # LOAD CONFIGS
    parser = argparse.ArgumentParser(description="Display training information from a YAML config file.")
    parser.add_argument("-c", "--config", default="config2.yaml", help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    EXPERIMENT_DIR = Path(config["experiment_dir"])

    SRC_DIRS = config["src_dirs"]
    DST_DIRS = config["dst_dirs"]

    encoder_path = config["model_paths"]["encoder"]
    src_decoder_path = config["model_paths"]["src_decoder"]
    dst_decoder_path = config["model_paths"]["dst_decoder"]
    inter_path = config["model_paths"]["inter"]
    g_optim_path = config["model_paths"]["g_optim"]
    dweight_path = config["model_paths"]["dweight"]
    d_grads_path = config["model_paths"]["d_grads"]
    d_optim_path = config["model_paths"]["d_optim"]

    steps = config["training"]["steps"]
    total_steps = config["training"]["total_steps"]
    total_steps = total_steps if total_steps > steps else 1e15

    RANDOM_WARP           = config["training"]["random_warp"]
    LEARNING_RATE_DROPOUT = config["training"]["learning_rate_dropout"]
    PERPETUAL_LOSS        = config["training"]["perpetual_loss"]
    GAN_POWER             = config["training"]["gan_power"]

    DTYPE_MAP = {
        "float32" : torch.float32,
        "float"   : torch.float32,
        "float16" : torch.float16,
        "half"    : torch.float16,
        "double"  : torch.float64,
        "float64" : torch.float64,
        "bfloat16": torch.bfloat16,
    }

    DEVICE      = torch.device(config["training"]["device"])
    ED_DTYPE    = DTYPE_MAP[config["dtype"]["ed_dtype"]]
    INTER_DTYPE = DTYPE_MAP[config["dtype"]["inter_dtype"]]
    PERP_DTYPE  = DTYPE_MAP[config["dtype"]["perp_dtype"]]
    DISC_DTYPE  = DTYPE_MAP[config["dtype"]["disc_dtype"]]
    
    BATCH_SIZE  = config["training"]["batch_size"]

    MODELS_DIR  = EXPERIMENT_DIR / "models"
    PREVIEW_DIR = EXPERIMENT_DIR / "preview"
    CONFIGS_DIR = EXPERIMENT_DIR / "configs"

    NUM_CHANNELS   = config["image"]["num_channels"]
    IN_IMAGE_SIZE  = config["image"]["in_size"]
    OUT_IMAGE_SIZE = config["image"]["out_size"]

    EXPERIMENT_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR    .mkdir(exist_ok=True, parents=True)
    PREVIEW_DIR   .mkdir(exist_ok=True, parents=True)
    CONFIGS_DIR   .mkdir(exist_ok=True, parents=True)

    save_config(config, str(CONFIGS_DIR/f"config_{int(time.time())}.yaml"))

    # MODELS
    AE_DIMS = config["model_dimensions"]["ae_dims"]
    E_DIMS  = config["model_dimensions"]["e_dims"]
    D_DIMS  = config["model_dimensions"]["d_dims"]
    M_DIMS  = config["model_dimensions"]["m_dims"]

    inter_in_features  = E_DIMS * 8 * ( IN_IMAGE_SIZE // 32) ** 2
    inter_out_features = E_DIMS * 8 * (OUT_IMAGE_SIZE // 32) ** 2

    model_path = MODELS_DIR/f"encoder_{steps//1000:03d}.pth"
    if (model_path).exists(): encoder_path = str(model_path)
    model_path = MODELS_DIR/f"src_decoder_{steps//1000:03d}.pth"
    if (model_path).exists(): src_decoder_path = str(model_path)
    model_path = MODELS_DIR/f"dst_decoder_{steps//1000:03d}.pth"
    if (model_path).exists(): dst_decoder_path = str(model_path)
    model_path = MODELS_DIR/f"inter_{steps//1000:03d}.pth"
    if (model_path).exists(): inter_path = str(model_path)
    model_path = MODELS_DIR/f"g_optim_{steps//1000:03d}.pth"
    if (model_path).exists(): g_optim_path = str(model_path)
    model_path = MODELS_DIR/f"dweight_{steps//1000:03d}.pth"
    if (model_path).exists(): dweight_path = str(model_path)
    model_path = MODELS_DIR/f"d_grads_{steps//1000:03d}.pth"
    if (model_path).exists(): d_grads_path = str(model_path)
    model_path = MODELS_DIR/f"d_optim_{steps//1000:03d}.pth"
    if (model_path).exists(): d_optim_path = str(model_path)

    # TRAINING INFO
    print(f"Experiment Directory: {EXPERIMENT_DIR}")
    print(f"  Models : {MODELS_DIR}")
    print(f"  Preview: {PREVIEW_DIR}")
    print(f"Source Directories:")
    for src in SRC_DIRS:
        print(f"  - {src}")
    print(f"Destination Directories:")
    for dst in DST_DIRS:
        print(f"  - {dst}")
    print(f"\nModels | (device:{DEVICE}):")
    print(f"  Encoder    : {encoder_path} ({ED_DTYPE})")
    print(f"  Src_decoder: {src_decoder_path} ({ED_DTYPE})")
    print(f"  Dst_decoder: {dst_decoder_path} ({ED_DTYPE})")
    print(f"  Inter      : {inter_path} ({INTER_DTYPE})")
    print(f"  DWeight    : {dweight_path} ({DISC_DTYPE })")
    print(f"  D Grads    : {d_grads_path} ({DISC_DTYPE })")
    print()
    print(f"  G optim: {g_optim_path}")
    print(f"  D optim: {d_optim_path}")
    
    print(f"\nTraining Parameters:")
    print(f"  Steps                : {steps}")
    print(f"  Total Steps          : {total_steps}")
    print(f"  Batch Size           : {BATCH_SIZE}")
    print(f"  Random Warp          : {RANDOM_WARP}")
    print(f"  Learning Rate Dropout: {LEARNING_RATE_DROPOUT}")
    print(f"  Perpetual Loss       : {PERPETUAL_LOSS} ({PERP_DTYPE})")
    print(f"  GAN Power            : {GAN_POWER}")

    print(f"\nImage Parameters:")
    print(f"  Number of Channels: {NUM_CHANNELS}")
    print(f"  Input Image Size: {IN_IMAGE_SIZE}")
    print(f"  Output Image Size: {OUT_IMAGE_SIZE}")
    print(f"\nModel Dimensions:")
    print(f"  AE Dims: {AE_DIMS}")
    print(f"  E Dims: {E_DIMS}")
    print(f"  D Dims: {D_DIMS}")
    print(f"  M Dims: {M_DIMS}")
    print(f"  Inter In Features: {inter_in_features}")
    print(f"  Inter Out Features: {inter_out_features}")
    print()

    # MODELS
    ENCODER = torch.load(encoder_path).to(device=DEVICE, dtype=ED_DTYPE) if encoder_path else \
        Encoder(E_DIMS).to(device=DEVICE, dtype=ED_DTYPE)
    # ENCODER = torch.compile(ENCODER, backend="eager")

    SRC_DECODER = torch.load(src_decoder_path).to(device=DEVICE, dtype=ED_DTYPE) if src_decoder_path else \
        Decoder(AE_DIMS * 4, D_DIMS, M_DIMS).to(device=DEVICE, dtype=ED_DTYPE)
    # SRC_DECODER = torch.compile(SRC_DECODER, backend="eager")

    DST_DECODER = torch.load(dst_decoder_path).to(device=DEVICE, dtype=ED_DTYPE) if dst_decoder_path else \
        Decoder(AE_DIMS * 4, D_DIMS, M_DIMS).to(device=DEVICE, dtype=ED_DTYPE)
    # DST_DECODER = torch.compile(DST_DECODER, backend="eager")
    
    INTER = torch.load(inter_path).to(device=DEVICE, dtype=INTER_DTYPE) if inter_path else \
        nn.Sequential(nn.Linear(inter_in_features, AE_DIMS), nn.Linear(AE_DIMS, inter_out_features)).to(device=DEVICE, dtype=INTER_DTYPE)
    # INTER = torch.compile(INTER, backend="eager")

    if GAN_POWER > 0.000001:
        DISCRIMINATOR = torch.load(dweight_path)
        disc_grads    = torch.load(d_grads_path)

        for name, param in DISCRIMINATOR.named_parameters():
            param.grad = disc_grads[name]
        
        DISCRIMINATOR = DISCRIMINATOR.to(device=DEVICE, dtype=DISC_DTYPE)
        # DISCRIMINATOR = torch.compile(DISCRIMINATOR, backend="eager")
    
    GAN_SMOOHTING = 0.1
    GAN_NOISE     = 0.0

    # LOSS + OPTIMIZERS
    mssim11_loss = MSSIM_Loss(data_range=1.0, size_average=True, win_size=11, channel=3)

    if PERPETUAL_LOSS:
        perceptual_loss = lpips.LPIPS(net=PERPETUAL_LOSS, lpips=True).to(device=DEVICE, dtype=PERP_DTYPE)

    lr_dropout = 0.3  if LEARNING_RATE_DROPOUT else 1.
    lr_cos     = 500. if LEARNING_RATE_DROPOUT else 0.
    
    g_optimizer = AdaBelief([
        *ENCODER.parameters(),
        *SRC_DECODER.parameters(),
        *DST_DECODER.parameters(),
        *INTER.parameters(),
        ], lr=5e-5, eps=1e-16, beta_1=0.9, beta_2=0.999, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=0.0)
    
    if g_optim_path:
        try:
            g_optimizer.load_state_dict(torch.load(g_optim_path))
        except ValueError as e:
            print(e)
            print("g optim not loaded")
    
    if GAN_POWER > 0:
        d_optimizer = AdaBelief(DISCRIMINATOR.parameters(), lr=5e-5, eps=1e-16, beta_1=0.9, beta_2=0.999, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=0.0)
        if d_optim_path:
            d_optimizer.load_state_dict(torch.load(d_optim_path))

    # DATA LOADERS
    dataset = SrcDstDataset(
        src_root_dirs=SRC_DIRS,
        dst_root_dirs=DST_DIRS,
        image_size=OUT_IMAGE_SIZE,
        src_rotation_range=[-10, 10],
        dst_rotation_range=[-10, 10],
        src_scale_range=[0.8, 1.2],
        dst_scale_range=[0.8, 1.2],
        src_random_flip=False,
        dst_random_flip=True,
        src_random_hue=0.25,
        dst_random_hue=0.25
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # PREVIEW SAMPLES
    preview_count = 8
    src_preview_image, src_preview_mask, dst_preview_image, dst_preview_mask = next(iter(loader))

    src_preview_image = TF.resize(src_preview_image[:preview_count], size=IN_IMAGE_SIZE)
    src_preview_mask  = TF.resize(src_preview_mask [:preview_count], size=IN_IMAGE_SIZE)
    dst_preview_image = TF.resize(dst_preview_image[:preview_count], size=IN_IMAGE_SIZE)
    dst_preview_mask  = TF.resize(dst_preview_mask [:preview_count], size=IN_IMAGE_SIZE)

    src_preview_image_warped = randomWarp(src_preview_image, 0.20).clamp(0, 1).to(device=DEVICE, dtype=ED_DTYPE)
    dst_preview_image_warped = randomWarp(dst_preview_image, 0.20).clamp(0, 1).to(device=DEVICE, dtype=ED_DTYPE)

    src_preview_image, src_preview_mask = src_preview_image.to(device=DEVICE, dtype=ED_DTYPE), src_preview_mask.to(device=DEVICE, dtype=ED_DTYPE)
    dst_preview_image, dst_preview_mask = dst_preview_image.to(device=DEVICE, dtype=ED_DTYPE), dst_preview_mask.to(device=DEVICE, dtype=ED_DTYPE)

    save_input_preview(
        src_preview=to_cpu(src_preview_image),
        src_preview_warped=to_cpu(src_preview_image_warped),
        src_preview_masked=to_cpu(torch.minimum(src_preview_mask, src_preview_image)),
        dst_preview=to_cpu(dst_preview_image),
        dst_preview_warped=to_cpu(dst_preview_image_warped),
        dst_preview_masked=to_cpu(torch.minimum(dst_preview_mask, dst_preview_image)),
        dir=str(EXPERIMENT_DIR)
        )

    # TRAIN
    LOSSES = {"steps": [], "src": [], "dst": []}

    if (EXPERIMENT_DIR/"losses.pkl").exists():
        with open(str(EXPERIMENT_DIR/"losses.pkl"), "rb") as file:
            LOSSES = pickle.load(file)
            LOSSES["steps"] = LOSSES["steps"][:steps]
            LOSSES["src"  ] = LOSSES["src"  ][:steps]
            LOSSES["dst"  ] = LOSSES["dst"  ][:steps]

    kernel_size = OUT_IMAGE_SIZE // 32
    if kernel_size % 2 == 0:
        kernel_size -= 1

    while steps < total_steps:
        for src_image, src_mask, dst_image, dst_mask in loader:
            START_TIME = int(time.time() * 1000)
            if steps >= total_steps:
                break
            steps += 1

            # DATA PREP
            if RANDOM_WARP:
                src_in = randomWarp(src_image, 0.20).clamp(0, 1).to(device=DEVICE, dtype=ED_DTYPE)
                dst_in = randomWarp(dst_image, 0.20).clamp(0, 1).to(device=DEVICE, dtype=ED_DTYPE)

            src_image = src_image.to(device=DEVICE, dtype=ED_DTYPE); 
            src_mask = src_mask.to(device=DEVICE, dtype=ED_DTYPE)
            src_mask_blur = TF.gaussian_blur(src_mask, kernel_size=kernel_size).clip(0, 0.5) * 2

            dst_image = dst_image.to(device=DEVICE, dtype=ED_DTYPE); 
            dst_mask = dst_mask.to(device=DEVICE, dtype=ED_DTYPE)
            dst_mask_blur = TF.gaussian_blur(dst_mask, kernel_size=kernel_size).clip(0, 0.5) * 2

            # FORWARD
            if IN_IMAGE_SIZE == OUT_IMAGE_SIZE:
                src_in = src_in if RANDOM_WARP else src_image
                dst_in = dst_in if RANDOM_WARP else dst_image
            else:
                src_in = TF.resize(src_in if RANDOM_WARP else src_image, size=IN_IMAGE_SIZE)
                dst_in = TF.resize(dst_in if RANDOM_WARP else dst_image, size=IN_IMAGE_SIZE)         

            src_code = INTER(ENCODER(src_in).to(dtype=INTER_DTYPE)).view(-1, E_DIMS * 8 * 4, OUT_IMAGE_SIZE//64, OUT_IMAGE_SIZE//64).to(dtype=ED_DTYPE)
            dst_code = INTER(ENCODER(dst_in).to(dtype=INTER_DTYPE)).view(-1, E_DIMS * 8 * 4, OUT_IMAGE_SIZE//64, OUT_IMAGE_SIZE//64).to(dtype=ED_DTYPE)

            src_image_pred, src_mask_pred = SRC_DECODER(F.interpolate(src_code, scale_factor=2, mode="bilinear", align_corners=False))
            dst_image_pred, dst_mask_pred = DST_DECODER(F.interpolate(dst_code, scale_factor=2, mode="bilinear", align_corners=False))

            src_image_masked     , dst_image_masked      = src_image      * src_mask_blur, dst_image      * dst_mask_blur
            src_image_pred_masked, dst_image_pred_masked = src_image_pred * src_mask_blur, dst_image_pred * dst_mask_blur

            # LOSS CALC
            src_image_loss_ssim = 10 * mssim11_loss(src_image_pred_masked, src_image_masked)
            src_image_loss_mse  = torch.mean(10 * torch.square(src_image_pred_masked - src_image_masked))
            src_mask_loss_mse   = torch.mean(10 * torch.square(src_mask_pred - src_mask))

            dst_image_loss_ssim = 10 * mssim11_loss(dst_image_pred_masked, dst_image_masked)
            dst_image_loss_mse  = torch.mean(10 * torch.square(dst_image_pred_masked - dst_image_masked))
            dst_mask_loss_mse   = torch.mean(10 * torch.square(dst_mask_pred - dst_mask))

            src_loss = src_image_loss_ssim + src_image_loss_mse + src_mask_loss_mse
            dst_loss = dst_image_loss_ssim + dst_image_loss_mse + dst_mask_loss_mse

            g_loss = src_loss + dst_loss

            if PERPETUAL_LOSS:
                g_loss += 2 * perceptual_loss(src_image_pred_masked.to(dtype=PERP_DTYPE), src_image_masked.to(dtype=PERP_DTYPE)).mean().to(dtype=ED_DTYPE)

            if GAN_POWER > 0:
                src_pred_d1, src_pred_d2 = DISCRIMINATOR(src_image_pred_masked.detach().to(dtype=DISC_DTYPE))
                src_pred_d1_zeros = get_smooth_noisy_labels(0, src_pred_d1, smoothing=GAN_SMOOHTING, noise=GAN_NOISE).to(device=DEVICE, dtype=DISC_DTYPE)
                src_pred_d2_zeros = get_smooth_noisy_labels(0, src_pred_d2, smoothing=GAN_SMOOHTING, noise=GAN_NOISE).to(device=DEVICE, dtype=DISC_DTYPE)
                
                src_d1, src_d2 = DISCRIMINATOR(src_image_masked.to(dtype=DISC_DTYPE))
                src_d1_ones = get_smooth_noisy_labels(1, src_d1, smoothing=GAN_SMOOHTING, noise=GAN_NOISE).to(device=DEVICE, dtype=DISC_DTYPE)
                src_d2_ones = get_smooth_noisy_labels(1, src_d2, smoothing=GAN_SMOOHTING, noise=GAN_NOISE).to(device=DEVICE, dtype=DISC_DTYPE)

                d_loss = DLoss(src_d1_ones, src_d1) + DLoss(src_pred_d1_zeros, src_pred_d1) + DLoss(src_d2_ones, src_d2) + DLoss(src_pred_d2_zeros, src_pred_d2)
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                src_pred_d1, src_pred_d2 = DISCRIMINATOR(src_image_pred_masked.to(dtype=DISC_DTYPE))
                src_pred_d1_ones = torch.ones_like(src_pred_d1, device=DEVICE, dtype=DISC_DTYPE)
                src_pred_d2_ones = torch.ones_like(src_pred_d2, device=DEVICE, dtype=DISC_DTYPE)

                g_loss += GAN_POWER * (DLoss(src_pred_d1_ones, src_pred_d1) + DLoss(src_pred_d2_ones, src_pred_d2))                
                g_loss += 0.000001 * total_variation_mse(src_image_pred)

                src_mask_anti_blur = 1 - src_mask_blur
                g_loss += 0.02 * torch.mean(10 * torch.square(src_image_pred * src_mask_anti_blur - src_image * src_mask_anti_blur))

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            LOSSES["steps"].append(steps)
            LOSSES["src"].append(src_loss.item())
            LOSSES["dst"].append(dst_loss.item())

            print(f"\rSTEP: {steps:06d} | {int(time.time() * 1000) - START_TIME:05d}ms | [{src_loss.item():06f}][{dst_loss.item():06f}]", end="")
            
            if steps % 100 == 0:
                with torch.no_grad():
                    src_code = ENCODER(src_preview_image_warped if RANDOM_WARP else src_preview_image).to(dtype=INTER_DTYPE)
                    dst_code = ENCODER(dst_preview_image_warped if RANDOM_WARP else dst_preview_image).to(dtype=INTER_DTYPE)

                    src_code = INTER(src_code).view(-1, E_DIMS * 8 * 4, OUT_IMAGE_SIZE//64, OUT_IMAGE_SIZE//64).to(dtype=ED_DTYPE)
                    dst_code = INTER(dst_code).view(-1, E_DIMS * 8 * 4, OUT_IMAGE_SIZE//64, OUT_IMAGE_SIZE//64).to(dtype=ED_DTYPE)

                    src_preview_pred, _ = SRC_DECODER(F.interpolate(src_code, scale_factor=2, mode="bilinear", align_corners=False))
                    dst_preview_pred, _ = DST_DECODER(F.interpolate(dst_code, scale_factor=2, mode="bilinear", align_corners=False))
                    swp_preview_pred, _ = SRC_DECODER(F.interpolate(dst_code, scale_factor=2, mode="bilinear", align_corners=False))

                    save_output_preview(
                        src_src=to_cpu(src_preview_pred.clamp(0,1)),
                        dst_dst=to_cpu(dst_preview_pred.clamp(0,1)),
                        src_dst=to_cpu(swp_preview_pred.clamp(0,1)),
                        it=steps,
                        dir=str(EXPERIMENT_DIR))
                    
                    save_loss_plot(losses=LOSSES, it=steps, dir=str(EXPERIMENT_DIR))

            if steps % 1000 == 0:
                torch.save(ENCODER, str(MODELS_DIR/f"encoder_{steps//1000:03d}.pth"))
                torch.save(SRC_DECODER, str(MODELS_DIR/f"src_decoder_{steps//1000:03d}.pth"))
                torch.save(DST_DECODER, str(MODELS_DIR/f"dst_decoder_{steps//1000:03d}.pth"))
                torch.save(INTER, str(MODELS_DIR/f"inter_{steps//1000:03d}.pth"))
                torch.save(g_optimizer.state_dict(), str(MODELS_DIR/f"g_optim_{steps//1000:03d}.pth"))
                
                if GAN_POWER > 0:
                    d_grads = {name: param.grad for name, param in DISCRIMINATOR.named_parameters()}
                    torch.save(DISCRIMINATOR, str(MODELS_DIR/f"dweight_{steps//1000:03d}.pth"))
                    torch.save(d_grads      , str(MODELS_DIR/f"d_grads_{steps//1000:03d}.pth"))

                    torch.save(d_optimizer.state_dict(), str(MODELS_DIR/f"d_optim_{steps//1000:03d}.pth"))

                with open(str(EXPERIMENT_DIR/"losses.pkl"), "wb") as file:
                    pickle.dump(LOSSES, file)


if __name__=="__main__":
    main()