import os
import json
import torch
import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from collections import defaultdict

# Define consistent scene class mapping
SCENE_LABELS = ['rider', 'bus', 'person', 'train', 'traffic sign', 'car', 'bike', 'traffic light', 'motor', 'truck']
SCENE2ID = {k: i for i, k in enumerate(SCENE_LABELS)}
ID2SCENE = {i: k for i, k in enumerate(SCENE_LABELS)}

class BDDDualTaskDataset(Dataset):
    def __init__(self, root_path, parition="train", input_size=224):
        self.root = os.path.join(root_path, parition + '/')
        self.input_size = input_size
        assert os.path.exists(self.root), f"Data directory {self.root} does not exist."
        print(f"Loading dataset from {self.root}")


        # Finding the mean and std for normalization
        vals = [[] for _ in range(3)]  # RGB channels
        self.samples = []

        self.labels = json.load(open(self.root + "labels.json", 'r'))
        self.labels = [item['label'] for item in self.labels]

        for fname in tqdm(os.listdir(self.root)):
            if fname.endswith('.jpg'):
                image = Image.open(os.path.join(self.root, fname))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                for channel in range(3):
                    vals[channel].extend(list(image.getdata(band=channel)))
                self.samples.append((image, self.labels[int(fname.split('.')[0])]))

        mean = [np.mean(vals[channel]) for channel in range(3)]
        std = [np.std(vals[channel]) for channel in range(3)]

        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        print("Total data available is", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        # Random rotation
        angle = random.uniform(0, 360)
        rotated = image.rotate(angle, resample=Image.BILINEAR)

        image_tensor = self.transform(image)
        image_rot_tensor = self.transform(rotated)

        return {"image": image_tensor, 
                "label": torch.tensor(label), 
                "image_rot": image_rot_tensor, 
                "angle": torch.tensor([angle / 360], dtype=torch.float32) # Normalized angle to [0, 1]
                }


def prepare_BDD(image_dir, label_dir, save_path, input_size=224, num_samples=5_000):
    """
    Prepares the BDD dataset by extracting images and labels, cropping them based on annotations,
    and saving the processed dataset to the specified path.
    """
    assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist."
    assert os.path.exists(label_dir), f"Label directory {label_dir} does not exist."
    print(f"Loading dataset images from {image_dir} and labels from {label_dir}...")
    # Finding the mean and std for normalization
    vals = [[] for _ in range(3)]  # RGB channels
    samples = []
    class_names = set()

    for fname in tqdm(os.listdir(image_dir)[5000:]):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            img_path = os.path.join(image_dir, fname)
            image = Image.open(img_path)
            # image.save(f"original_{fname}.jpg")

            for channel in range(3):
                vals[channel].extend(list(image.getdata(band=channel)))
            
            label_path = os.path.join(label_dir, fname + ".json")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    data = json.load(f)
                    count = 0
                    for obj in data.get("objects", []):
                        if obj["geometryType"] != "rectangle" or obj["classTitle"] not in SCENE_LABELS:
                            continue

                        class_name = obj["classTitle"]
                        x1, y1 = obj["points"]["exterior"][0]
                        x2, y2 = obj["points"]["exterior"][1]

                        # Compute full rectangle corners
                        x_min, y_min = min(x1, x2), min(y1, y2)
                        x_max, y_max = max(x1, x2), max(y1, y2)

                        if x_max - x_min < input_size // 2 or y_max - y_min < input_size // 2:
                            continue
                        # Crop
                        crop = image.crop((x_min, y_min, x_max, y_max))
                        # print(crop.size)
                        # crop.save(f"{count}_{fname}")
                        class_names.add(class_name)
                        samples.append((crop, SCENE2ID[class_name]))
                        count += 1
        if len(samples) == num_samples:
            print(f"Reached the limit of {num_samples} samples.")
            break
       
    print(f"Found {len(class_names)} unique classes: {class_names} and a total of {len(samples)} samples.")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Saving dataset
    labels = []
    for idx, (image, label) in enumerate(samples):
        image.save(os.path.join(save_path, f"{idx}.jpg"))
        labels.append({
            "filename": f"{idx}.jpg",
            "label": label
        })
    with open(os.path.join(save_path, "labels.json"), 'w') as f:
        json.dump(labels, f, indent=4)
    print(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    parition = 'val'  # or 'val', 'test'

    # Note: Test set of BDD-100k doesn't have annotations, so we will use the second half of validation set for testing.
    prepare_BDD(
        image_dir=f'/scr/Pedram/VisualLearning/bdd100k/bdd100k-images/{parition}/img/',
        label_dir=f'/scr/Pedram/VisualLearning/bdd100k/bdd100k-images/{parition}/ann/',
        save_path=f'/scr/Pedram/VisualLearning/processed_bdd100k/{parition}',
        input_size=224,
        num_samples=5_000,
    )