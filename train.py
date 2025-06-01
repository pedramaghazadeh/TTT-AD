import wandb

import torch.nn as nn

from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    cls_loss_fn = nn.CrossEntropyLoss()
    rot_loss_fn = nn.MSELoss()
    
    total_cls_loss, total_rot_loss = 0, 0

    for batch in tqdm(dataloader):
        x, label, angle = batch["image"].to(device), batch["label"].to(device), batch["angle"].to(device)

        optimizer.zero_grad()
        
        cls_logits, rot_pred = model(x)
        
        cls_loss = cls_loss_fn(cls_logits, label)
        rot_loss = rot_loss_fn(rot_pred, angle)
        
        loss = cls_loss + 0.1 * rot_loss  # Weighted sum
        loss.backward()
        
        optimizer.step()
        
        total_cls_loss += cls_loss.item()
        total_rot_loss += rot_loss.item()
        wandb.log({"cls_loss": cls_loss.item(), "rot_loss": rot_loss.item()})

    return total_cls_loss / len(dataloader), total_rot_loss / len(dataloader)