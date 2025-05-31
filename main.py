import argparse

from torch.utils.data import DataLoader
from model import *
from train import *
from dataloader import *
from datasets import load_dataset



def train_model(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetTTT(num_classes=10).to(device)

    dataset = BDDDualTaskDataset(root=args.data_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        cls_loss, rot_loss = train_epoch(model, loader, optimizer, device)
        print(f"[Epoch {epoch}] Classification Loss: {cls_loss:.4f} | Rotation Loss: {rot_loss:.4f}")

if __name__ == "__main__":
    # Check if the dataset is available
    ds = load_dataset("dgural/bdd100k")
    ds.save_to_disk("bdd100k_dataset")


    parser = argparse.ArgumentParser(description="Train EfficientNet for TTT")
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--data-path', type=str, default='/path/to/classification/images', help='Path to the dataset')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Dataset not found at {args.data_path}. Please provide a valid path.")
        exit(1)

    train_model(args)