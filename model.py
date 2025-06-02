import torch
import torch.nn as nn
import timm

class EfficientNetTTT(nn.Module):
    def __init__(self, num_classes, base='efficientnet_b0', first_n_layers=None):
        super().__init__()
        self.encoder = timm.create_model(base, pretrained=True, num_classes=0, global_pool='avg')
        self.feature_dim = self.encoder.num_features

        self.class_head = nn.Linear(self.feature_dim, num_classes)
        if first_n_layers is not None:
            self.encoder_rotation = nn.Sequential(*list(self.encoder.children())[:first_n_layers])
            # Last layer is global pool, so we need to flatten the features
            self.encoder_rotation.add_module('flatten', nn.Flatten())
            self.feature_dim_rot = self.encoder_rotation[-1].num_features
            # Rotation head for the first n layers
            self.rotation_head = nn.Sequential(
                nn.Linear(self.feature_dim_rot, 128),
                nn.ReLU(),
                nn.Linear(128, 1)  # Regression: rotation in degrees
            )
        else:
            self.encoder_rotation = self.encoder
            self.rotation_head = nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)  # Regression: rotation in degrees
            )

    def forward(self, x):
        feats = self.encoder(x)
        cls_logits = self.class_head(feats)

        feats_rot = self.encoder_rotation(x)
        rot_pred = self.rotation_head(feats_rot)
        return cls_logits, rot_pred

# def extractor_from_layer4(net):
# 	layers = [net.conv1, net.bn1, net.relu, net.maxpool,
# 				 net.layer1, net.layer2, net.layer3, net.layer4, 
# 					net.avgpool, ViewFlatten()]
# 	return nn.Sequential(*layers)