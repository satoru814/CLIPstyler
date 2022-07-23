import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import utils
from models.encoder import CLIPModel

from itertools import chain
import matplotlib as plt
import numpy as np
import clip
import wandb
from config import CFG
import argparse

from PIL import Image
import PIL

from torchvision import transforms, models


import StyleNet

class CLIP():
    def __init__(self, args):
        self.wandb = args.wandb
        self.save_weight = args.save_weight
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        self.lr = CFG.lr
        self.content_path = CFG.content_path
        self.text = CFG.text
        self.source = CFG.source
        self.crop_n = CFG.crop_n
        self.step = CFG.step
        self.patch_threshold = CFG.patch_threshold

    def build_model(self):
        self.style_net = StyleNet.UNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.style_net.parameters(), self.lr)
        self.VGG = models.vgg19(pretrained=True).features
        self.VGG.to(self.deivice)

    def train(self):
        #wandb
        if self.wandb:
            run = wandb.init(**CFG.wandb, settings=wandb.Settings(code_dir="."))
            wandb.watch(models=(self.Net), log_freq=100)
        content_features = utils.get_features(content, self.VGG) #normalize

        clip_model = clip.load("ViT-B/32", self.device, jit=False)
        tokens_features = clip.tokenize(self.text).to(self.device)
        text_features = clip_model.encode_text(tokens_features).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features = text_features.norm(dim=-1, keepdim=True)

        tokens_source = clip.tokenize(self.text).to(self.device)
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source = text_source.norm(dim=-1, keepdim=True)

        c_loss = nn.MSELoss()
        random_patching  = utils.get_transforms()
        for step in range(CFG.step):
            content_loss = 0
            patch_loss = 0

            self.Net.train()
            losses = {"cross_entropy" : 0}
            print("step",step)
            target = self.style_net(self.content_image).to(self.device)
            target_features = utils.get_features(target, self.VGG) #normalize
            content_loss += c_loss(target_features["conv4_2"], content_features["conv4_2"])
            content_loss += c_loss(target_features["conv5_2"], content_features["conv5_2"])

            img_proc = []
            for n in range(self.crop_n):
                temp_patch = random_patching(content)
                img_proc.append(temp_patch)
            img_proc = torch.cat(img_proc, dim=0).to(self.device)
            img_features = clip_model.encode_image(img_proc) #normalize
            img_features /= img_features.norm(dim=-1, keepdim=True)
            img_direction = img_features - content_features
            img_direction /= img_direction.norm(dim=-1, keepdim=True)
            
            loss_temp = 1 - torch.cosine_similarity(img_direction, text_direction, dim=1)
            loss_temp[loss_temp <= self.patch_threshold]
            text_direction = (text_features - text_source).repeat(img_direction.size(0), 1)
            
            if self.wandb:
                wandb.log(losses)

        self.save()

        if self.wandb:
            run.finish()


    def inference(self, query=CFG.test_query ,weight=None):
        if weight:
            self.Net.load(weight)
        img_embeddings = []
        filenames = []
        self.Net.eval()
        with torch.no_grad():
            for i, item in enumerate(self.val_loader):
                img = item[0].to(self.device).float()
                filename = item[4]
                img_features = self.Net.image_encoder(img)
                img_embedding = self.Net.image_projection(img_features)
                img_embeddings.append(img_embedding)
                filenames += filename
            img_embeddings = torch.cat(img_embeddings)
        print("find_match call")
        utils.find_matches(self.Net, query, img_embeddings, filenames, self.device)


    def save(self):
        torch.save(self.Net.state_dict(), CFG.MODEL_SAVE_PATH)