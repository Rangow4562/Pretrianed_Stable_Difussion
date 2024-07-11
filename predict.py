#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torch
from model import BiSeNet
from scipy.ndimage.filters import gaussian_filter
from face_detection import FaceDetection
from trimap_class import trimap , Erosion
import os, sys
from refine_mask import ImageProcessing
# from pymatting import *

class SemanticSegmentationModel:
    def __init__(self, model_path="./model_final_diss.pth" , gpu=False):
        self.n_classes = 19
        self.size = 10        
        self.scale = 1.0
        self.net = BiSeNet(n_classes=self.n_classes)
        self.gpu = gpu
        if self.gpu:
            self.net.cuda()
            self.net.load_state_dict(torch.load(model_path))
        else:    
            self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.net.eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.refine_process = ImageProcessing()

    def resize_target_resolution(self, image, target_size):
        width, height = image.size
        aspect_ratio = width / height

        if width > height:
            new_width = min(width, target_size[0])
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(height, target_size[1])
            new_width = int(new_height * aspect_ratio)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image
    
    def erode_and_dilate(self, mask, k_size, iterations):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
        eroded = cv2.erode(mask, kernel, iterations=iterations)
        dilated = cv2.dilate(mask, kernel, iterations=iterations)
        trimap = np.full(mask.shape, 128)
        trimap[eroded >= 254] = 255
        trimap[dilated <= 1] = 0
        return trimap

    def generate_trimap(self, mask, threshold=0.05, iterations=3):
        threshold = threshold * 255
        trimap = mask.copy()
        trimap = trimap.astype("uint8")
        trimap = self.erode_and_dilate(trimap, k_size=(7, 7), iterations=iterations)
        return trimap
    
    def process_face_image(self, im, parsing_anno, stride=1):
        im = np.array(im)
        mask_bg = im.copy().astype(np.uint8)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(
            vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST
        )
        parts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]
        face_masks = []
        for i in parts:
            vis_parsing_anno_bg = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]), dtype=np.uint8)  # Removed the alpha channel
            face_index = np.where(vis_parsing_anno == i)
            vis_parsing_anno_bg[face_index] = 255  # Assigning full opacity
            face_masks.append(vis_parsing_anno_bg)
        
        face_masks_combined = np.logical_or.reduce(face_masks).astype(np.uint8) * 255  # Convert to uint8
        face_masks_combined = gaussian_filter(face_masks_combined, sigma=2)
        trimap_image = self.generate_trimap(face_masks_combined)
        mask_bg = Image.fromarray(mask_bg)
        trimap_image = Image.fromarray(trimap_image)
        input_img = self.refine_process.get_data(mask_bg, trimap_image)
        alpha = self.refine_process.infer_one_image(input_img)
        cutout = self.refine_process.cal_foreground(mask_bg,alpha)
        return cutout
    
    def cv2_to_pil(self,image):
        assert image.dtype in [np.uint8, np.float32, np.float64]
        if image.dtype in [np.float32, np.float64]:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(image)
        
    def _resize_pil_image(self, image, size, resample="bicubic"):
        filters = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "box": Image.BOX,
        "hamming": Image.HAMMING,
        "lanczos": Image.LANCZOS,
        "nearest": Image.NEAREST,
        "none": Image.NEAREST,
        }

        if not isiterable(size):
            size = (int(image.width * size), int(image.height * size))
        image = image.resize(size, filters[resample.lower()])
        return image

    def load_image(self, image, mode=None, size=None, resample="box"):
        if mode is not None:
            mode = mode.upper()
            mode = "L" if mode == "GRAY" else mode
            image = image.convert(mode)
        if size is not None:
            image = self._resize_pil_image(image, size, resample)
        image = np.array(image) / 255.0
        return image
        
    def infer(self, img ,target_size=(512, 512)):
        image = self.resize_target_resolution(img, target_size)
        if self.gpu:
            img_tensor = self.to_tensor(image).unsqueeze(0).cuda()
        else:
            img_tensor = self.to_tensor(image)    
            img_tensor = torch.unsqueeze(img_tensor, 0)
            
        with torch.no_grad():
            out = self.net(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

        vis_im = self.process_face_image(image, parsing)
        return vis_im
         
if __name__ == "__main__":
    model = SemanticSegmentationModel(model_path="79999_iter.pth")
    face_model = FaceDetection()
    refine_process = ImageProcessing()
    img = Image.open("1.png")
    faces = face_model.detect_face(img)
    print("Number of faces detected:", len(faces))
    if len(faces) >= 2 :
        raise Exception(f"Number of faces expected:", len(faces))
    vis_im = model.infer(img=img)
    vis_im.save("vis_im.png", format="PNG")


