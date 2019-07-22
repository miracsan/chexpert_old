# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:17:34 2019

@author: Mirac
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from chexpertDataset import CheXpertDataset

### CLASS ACTIVATION ####
class HookModel(nn.Module):
    def __init__(self, model):
        super(HookModel, self).__init__()
        self.gradients = None
        self.features = model.features
        self.classifier = model.classifier

        
    def activations_hook(self, grad):
        self.gradients = grad
        
        
    def forward(self, x):
        ## your forward pass
        x = self.features(x)
        h = x.register_hook(self.activations_hook)
        out = F.relu(x, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        return out

    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features(x)


def create_dataloader_cam(PATH_TO_MAIN_FOLDER = "/content", UNCERTAINTY = 'weighted_multiclass'):
  # use imagenet mean,std for normalization
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]


  N_LABELS = 14  # we are predicting 14 labels

  BATCH_SIZE = 1
  data_transforms = {
          'train': transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.Scale(224),
              # because scale doesn't always give 224 x 224, this ensures 224 x
              # 224
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize(mean, std)
          ]),
          'valid': transforms.Compose([
              transforms.Scale(224),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize(mean, std)
          ]),
      }

  # create train/val dataloaders
  transformed_datasets = {}
  transformed_datasets['train'] = CheXpertDataset(
      path_to_main_folder=PATH_TO_MAIN_FOLDER,
      fold='train',
      transform=data_transforms['train'],
      uncertainty=UNCERTAINTY)
  transformed_datasets['valid'] = CheXpertDataset(
      path_to_main_folder=PATH_TO_MAIN_FOLDER,
      fold='valid',
      transform=data_transforms['valid'],
      uncertainty=UNCERTAINTY)

  dataloaders = {}
  dataloaders['train'] = torch.utils.data.DataLoader(
      transformed_datasets['train'],
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=8)
  dataloaders['valid'] = torch.utils.data.DataLoader(
      transformed_datasets['valid'],
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=8)
  
  return dataloaders

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])