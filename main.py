import argparse
import wandb

from torch.utils.data import DataLoader
from model import *
from train import *
from data import *



def train_model(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetTTT(num_classes=10).to(device)
    wandb.init(project="ttt-ad", name="ttt-ad-training", config=args)
    wandb.watch(model, log="all")

    dataset = BDDDualTaskDataset(root_path=args.data_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        cls_loss, rot_loss = train_epoch(model, loader, optimizer, device)
        print(f"[Epoch {epoch}] Classification Loss: {cls_loss:.4f} | Rotation Loss: {rot_loss:.4f}")
        wandb.log({"epoch": epoch, "cls_loss": cls_loss, "rot_loss": rot_loss})

    # torch.save(model.state_dict(), f"model_.pth")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet for TTT")
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--data-path', type=str, default='/scr/Pedram/VisualLearning/processed_bdd100k/train/', help='Path to the dataset')

    args = parser.parse_args()

    assert os.path.exists(args.data_path), f"Dataset path {args.data_path} does not exist."
    train_model(args)