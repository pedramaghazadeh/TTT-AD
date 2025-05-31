import os
import json
import torch
import random

import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

from torchvision import transforms as T


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    cls_loss_fn = nn.CrossEntropyLoss()
    rot_loss_fn = nn.MSELoss()

    total_cls_loss, total_rot_loss = 0, 0

    for x, label, angle in dataloader:
        x, label, angle = x.to(device), label.to(device), angle.to(device)
        optimizer.zero_grad()
        cls_logits, rot_pred = model(x)
        cls_loss = cls_loss_fn(cls_logits, label)
        rot_loss = rot_loss_fn(rot_pred, angle)
        loss = cls_loss + 0.1 * rot_loss  # Weighted sum
        loss.backward()
        optimizer.step()
        total_cls_loss += cls_loss.item()
        total_rot_loss += rot_loss.item()

    return total_cls_loss / len(dataloader), total_rot_loss / len(dataloader)