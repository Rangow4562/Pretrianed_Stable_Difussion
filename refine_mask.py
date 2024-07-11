import os
from PIL import Image
from os.path import join as opj
from torchvision.transforms import functional as F
from detectron2.engine import default_argument_parser
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

class ImageProcessing:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.init_model(model='vitmatte-s', checkpoint='./ViTMatte_S_Com.pth', device=device)
        
    def infer_one_image(self, image):
        output = self.model(image)['phas'].flatten(0, 2)
        output = F.to_pil_image(output)
        return output

    def init_model(self, model, checkpoint, device):
        assert model in ['vitmatte-s', 'vitmatte-b']
        if model == 'vitmatte-s':
            config = 'ViTMatte/configs/common/model.py'
            cfg = LazyConfig.load(config)
            model = instantiate(cfg.model)
            model.to(device)
            model.eval()
            DetectionCheckpointer(model).load(checkpoint)
        elif model == 'vitmatte-b':
            config = 'ViTMatte/configs/common/model.py'
            cfg = LazyConfig.load(config)
            cfg.model.backbone.embed_dim = 768
            cfg.model.backbone.num_heads = 12
            cfg.model.decoder.in_chans = 768
            model = instantiate(cfg.model)
            model.to(device)
            model.eval()
            DetectionCheckpointer(model).load(checkpoint)
        return model

    def get_data(self, image, trimap):
        image = image.convert('RGB')
        image = F.to_tensor(image).unsqueeze(0)
        trimap = trimap.convert('L')
        trimap = F.to_tensor(trimap).unsqueeze(0)

        return {
            'image': image,
            'trimap': trimap
        }

    def cal_foreground(self, image, alpha):
        image = image.convert('RGB')
        alpha = alpha.convert('L')
        alpha_image = alpha.copy()
        alpha = F.to_tensor(alpha).unsqueeze(0)
        image = F.to_tensor(image).unsqueeze(0)
        foreground = image * alpha + (1 - alpha)
        foreground = foreground.squeeze(0).permute(1, 2, 0).numpy()
        foreground = (foreground * 255).astype(np.uint8)
        foreground_image = Image.fromarray(foreground.astype(np.uint8), 'RGB')
        foreground_image.putalpha(alpha_image)
        return foreground_image

    def merge_new_bg(self, image_dir, bg_dir, alpha_dir):
        image = Image.open(image_dir).convert('RGB')
        bg = Image.open(bg_dir).convert('RGB')
        alpha = Image.open(alpha_dir).convert('L')
        image = F.to_tensor(image)
        bg = F.to_tensor(bg)
        bg = F.resize(bg, image.shape[-2:])
        alpha = F.to_tensor(alpha)
        new_image = image * alpha + bg * (1 - alpha)
        new_image = new_image.squeeze(0).permute(1, 2, 0).numpy()
        return new_image


# process = ImageProcessing()
# image_dir = 'mask_bg.png'
# trimap_dir = 'trimap_image_delt.png'
# bg_dir = './ViTMatte/demo/new_bg.jpg'
# image = Image.open(image_dir)
# trimap = Image.open(trimap_dir)

# input_img = process.get_data(image, trimap)
# alpha = process.infer_one_image(input_img)
# fg = process.cal_foreground(image,alpha)
# fg.save("fg.png")