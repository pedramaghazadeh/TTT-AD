import wandb
import torch

import torch.nn as nn
import numpy as np

from PIL import Image
from tqdm import tqdm

train_step, val_step, test_step, ttt_step = 0, 0, 0, 0

def train_epoch(model, dataloader, optimizer, device, ttt=True):
    model.train()
    cls_loss_fn = nn.CrossEntropyLoss()

    # rot_loss_fn = nn.MSELoss()
    rot_loss_fn = nn.CrossEntropyLoss()

    total_cls_loss, total_rot_loss = 0, 0
    total_acc = 0
    global train_step
    for batch in dataloader:
        x, label, angle = batch["image"].to(device), batch["label"].to(device), batch["angle"].to(device)

        optimizer.zero_grad()
        
        cls_logits, rot_pred = model(x)
        
        cls_loss = cls_loss_fn(cls_logits, label)
        if ttt:
            rot_loss = rot_loss_fn(rot_pred, angle)
            loss = cls_loss.mean() + rot_loss.mean()
        else:
            loss = cls_loss.mean()

        loss.backward()
        optimizer.step()
        
        total_cls_loss += cls_loss.item()
        if ttt:
            total_rot_loss += rot_loss.item()
            wandb.log({"train/cls_loss": cls_loss.item(), "train/rot_loss": rot_loss.item(), "train/step": train_step})
        else:
            wandb.log({"train/cls_loss": cls_loss.item(), "train/step": train_step})
        
        accuracy = (cls_logits.argmax(dim=1) == label).float().mean().item()
        wandb.log({"train/accuracy": accuracy, "train/step": train_step})
        train_step += 1

        total_acc += accuracy

    total_acc /= len(dataloader)
    wandb.log({"train/total_accuracy": total_acc})
    return total_cls_loss / len(dataloader), total_rot_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.eval()
    cls_loss_fn = nn.CrossEntropyLoss()
    # rot_loss_fn = nn.MSELoss()
    rot_loss_fn = nn.CrossEntropyLoss()

    total_cls_loss, total_rot_loss = 0, 0
    total_acc = 0
    global val_step
    with torch.no_grad():
        for batch in dataloader:
            x, label, angle = batch["image"].to(device), batch["label"].to(device), batch["angle"].to(device)

            cls_logits, rot_pred = model(x)

            cls_loss = cls_loss_fn(cls_logits, label)
            rot_loss = rot_loss_fn(rot_pred, angle)

            total_cls_loss += cls_loss.item()
            total_rot_loss += rot_loss.item()
            accuracy = (cls_logits.argmax(dim=1) == label).float().mean().item()
            total_acc += accuracy

            wandb.log({"val/cls_loss": cls_loss.item(), "val/rot_loss": rot_loss.item(), "val/step": val_step})
            val_step += 1

    total_acc /= len(dataloader)
    wandb.log({"val/total_accuracy": total_acc})
    return total_cls_loss / len(dataloader), total_rot_loss / len(dataloader)

def test_epoch(model, dataloader, device):
    model.eval()
    cls_loss_fn = nn.CrossEntropyLoss()
    # rot_loss_fn = nn.MSELoss()
    rot_loss_fn = nn.CrossEntropyLoss()

    total_cls_loss, total_rot_loss = 0, 0
    total_acc = 0
    global test_step
    with torch.no_grad():
        for batch in dataloader:
            x, label, angle = batch["image"].to(device), batch["label"].to(device), batch["angle"].to(device)

            cls_logits, rot_pred = model(x)

            cls_loss = cls_loss_fn(cls_logits, label)
            rot_loss = rot_loss_fn(rot_pred, angle)

            total_cls_loss += cls_loss.item()
            total_rot_loss += rot_loss.item()
            
            accuracy = (cls_logits.argmax(dim=1) == label).float().mean().item()
            wandb.log({"test/accuracy": accuracy, "test/step": test_step})
            test_step += 1
            total_acc += accuracy
    
    total_acc /= len(dataloader)
    wandb.log({"test/total_accuracy": total_acc, "test/step": test_step})
    test_step += 1
    return total_cls_loss / len(dataloader), total_rot_loss / len(dataloader)

def test_time_training_inference(model, dataloader, device, num_iterations=100):
    # Train the model on this batch of data
    ttt_model.train()
    acc = 0
    optimizer = torch.optim.Adam(ttt_model.parameters(), lr=1e-4)
    # rot_loss_fn = nn.MSELoss()
    rot_loss_fn = nn.CrossEntropyLoss()

    global ttt_step
    for batch in dataloader:
        # Batch size of 1
        ttt_model = model.copy()
        ttt_model.to(device)

        image, label, _, __ = batch["image"].to(device), batch["label"].to(device), batch["image_rot"], batch["angle"]
        # Rotate the image to create multiple views

        image = Image.fromarray(image.cpu().numpy().astype(np.uint8))
        # Create 4 rotated versions of the image  
        rotations = [0, 90, 180, 270]
        images = [image.rotate(angle) for angle in rotations]
        
        image_tensors = torch.stack([torch.tensor(np.array(img).transpose(2, 0, 1), dtype=torch.float32) for img in images]).to(device)
        rotation_labels = torch.tensor(rotations / 360, dtype=torch.float32).to(device)
        
        # Train the model
        for _ in range(num_iterations):
            optimizer.zero_grad()
            cls_logits, rot_pred = ttt_model(image_tensors)
            rot_loss = rot_loss_fn(rot_pred, rotation_labels)
            rot_loss.backward()
            optimizer.step()

        # Predict the output for the original image
        ttt_model.eval()
        with torch.no_grad():
            cls_logits, rot_pred = ttt_model(image.to(device).unsqueeze(0))
            class_pred = cls_logits.argmax(dim=1)
            acc += (class_pred == label).float().mean().item()

    wandb.log({"ttt/accuracy": acc / len(dataloader), "ttt/step": ttt_step})
    ttt_step += 1
    return acc / len(dataloader)