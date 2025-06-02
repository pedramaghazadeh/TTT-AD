import argparse
import wandb

from torch.utils.data import DataLoader
from model import *
from train import *
from data import *

CORRUPTED = ["blur", "fog", "rain", "snow", "sun_flare", "test"]

def train_model(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetTTT(num_classes=10).to(device)
    wandb.init(project="ttt-ad", name="ttt-ad-training", config=args)
    wandb.watch(model, log="all")

    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/*", step_metric="val/step")
    wandb.define_metric("test/*", step_metric="test/step")
    wandb.define_metric("ttt/*", step_metric="ttt/step")

    # dataset = BDDDualTaskDataset(root_path=args.data_path, partition="train")
    # train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=32)

    # # Validation dataset
    # val_dataset = BDDDualTaskDataset(root_path=args.data_path, partition="val")
    # val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=32)
    # # Test dataset
    # test_dataset = BDDDualTaskDataset(root_path=args.data_path, partition="test")
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=32)

    # print(f"Train dataset size: {len(dataset)}")
    # print(f"Validation dataset size: {len(val_dataset)}")
    # print(f"Test dataset size: {len(test_dataset)}")

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # for _ in tqdm(range(args.num_epochs), desc="Training and validating"):
    #     # Training
    #     train_epoch(model, train_dataloader, optimizer, device)
    #     # Validation
    #     validate_epoch(model, val_dataloader, device)

    # Test set
    # test_epoch(model, test_dataloader, device)

    # TTT on test dataset
    for partition in CORRUPTED:
        corrupted_dataset = BDDDualTaskDataset(root_path=args.data_path, partition=partition)
        corrupted_dataloader = DataLoader(corrupted_dataset, batch_size=1, shuffle=False, num_workers=32)
        test_time_training_inference(model, corrupted_dataloader, device)
    # torch.save(model.state_dict(), f"model_.pth")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet for TTT")
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--data-path', type=str, default='/scr/Pedram/VisualLearning/processed_bdd100k/', help='Path to the dataset')
    parser.add_argument('--num-epochs', type=int, default=2, help='Number of training epochs')

    args = parser.parse_args()

    assert os.path.exists(args.data_path), f"Dataset path {args.data_path} does not exist."
    train_model(args)