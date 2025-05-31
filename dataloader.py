import os
import json
import torch
import random
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from collections import defaultdict

# Define consistent scene class mapping
SCENE_LABELS = [
    'highway', 'residential', 'city street', 'parking lot', 'tunnel', 'gas stations'
]
SCENE2ID = {k: i for i, k in enumerate(SCENE_LABELS)}

class BDDDualTaskDataset(Dataset):
    def __init__(self, image_dir, label_json, input_size=224):
        self.image_dir = image_dir
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.samples = []

        with open(label_json, 'r') as f:
            data = json.load(f)
            for item in data:
                fname = item['name']
                scene = item.get('attributes', {}).get('scene', '')
                if scene not in SCENE2ID:
                    continue
                full_path = os.path.join(image_dir, fname)
                if os.path.exists(full_path):
                    self.samples.append((full_path, SCENE2ID[scene]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')

        # Random rotation
        angle = random.uniform(0, 360)
        rotated = image.rotate(angle, resample=Image.BILINEAR)
        image_tensor = self.transform(rotated)

        return image_tensor, torch.tensor(label), torch.tensor([angle], dtype=torch.float32)
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    trainset = BDDDualTaskDataset(
        image_dir='bdd100k/images/10k/train',
        label_json='bdd100k/labels/bdd100k_labels_images_train.json'
    )
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

    for batch in trainloader:
        images, class_labels, rotation_angles = batch
        print(images.shape)  # [B, 3, 224, 224]
        print(class_labels)  # [B]
        print(rotation_angles)  # [B, 1]
        break