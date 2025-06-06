import torch
import torch.nn as nn
import timm

def extract_layers(net, first_n_layers):
    print(f"Extracting layers up to: {first_n_layers}")
    layers = [net.conv_stem]
    for i in range(first_n_layers):
        layers.append(net.blocks[i])
    return nn.Sequential(*layers)

class EfficientNetTTT(nn.Module):
    def __init__(self, num_classes, base='efficientnet_b0', first_n_layers=None, ssl='square-rot'):
        super().__init__()
        self.encoder = timm.create_model(base, pretrained=True, num_classes=0, global_pool='avg')
        self.feature_dim = self.encoder.num_features
        self.ssl = ssl
        self.class_head = nn.Linear(self.feature_dim, num_classes)
        self.first_n_layers = first_n_layers

        if first_n_layers is not None:
            
            self.encoder_rotation = extract_layers(self.encoder, first_n_layers)
            # Last layer is global pool, so we need to flatten the features
            self.encoder_rotation.add_module('flatten', nn.Flatten())
            # Rotation head for the first n layers
            if self.ssl == 'rot':
                self.rotation_head = nn.Sequential(
                    nn.LazyLinear(128),
                    nn.ReLU(),
                    nn.Linear(128, 1)  # Regression: predicting angle in radians
                )
            elif self.ssl == 'square-rot':
                self.rotation_head = nn.Sequential(
                    nn.LazyLinear(128),
                    nn.ReLU(),
                    nn.Linear(128, 4)  # Classification: 4 classes for 0, 90, 180, 270 degrees
                )
            else:
                raise ValueError("Unsupported SSL type. Use 'rot' or 'square-rot'.")
        else:
            if self.ssl == 'rot':
                self.rotation_head = nn.Sequential(
                    nn.Linear(self.feature_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)  # Regression: predicting angle in radians
                )
            elif self.ssl == 'square-rot':
                # Rotation head for the full model
                self.rotation_head = nn.Sequential(
                    nn.Linear(self.feature_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 4), # Classification: 4 classes for 0, 90, 180, 270 degrees
                )

    def forward(self, x):
        feats = self.encoder(x)
        cls_logits = self.class_head(feats)

        if self.first_n_layers is not None:
            rot_pred = self.rotation_head(self.encoder_rotation(x))
        else:
            rot_pred = self.rotation_head(feats)

        return cls_logits, rot_pred