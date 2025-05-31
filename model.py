import torch
import torch.nn as nn
import timm

class EfficientNetTTT(nn.Module):
    def __init__(self, num_classes, base='efficientnet_b0'):
        super().__init__()
        self.encoder = timm.create_model(base, pretrained=True, num_classes=0, global_pool='avg')
        self.feature_dim = self.encoder.num_features

        self.class_head = nn.Linear(self.feature_dim, num_classes)
        self.rotation_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Regression: rotation in degrees
        )

    def forward(self, x):
        feats = self.encoder(x)
        cls_logits = self.class_head(feats)
        rot_pred = self.rotation_head(feats)
        return cls_logits, rot_pred