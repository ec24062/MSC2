# train_diffsep.py

import os
import argparse
import torch
import importlib.util
from torch.utils.data import DataLoader
from bleed_dataset import BleedDataset

def import_score_model(repo_dir: str):
    """
    Directly imports ScoreModelNCSNpp from score_models.py in repo_dir/models.
    """
    model_file = os.path.join(repo_dir, "models", "score_models.py")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"{model_file} not found")

    spec = importlib.util.spec_from_file_location("score_models", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ScoreModelNCSNpp

def parse_args():
    p = argparse.ArgumentParser(description="Train ScoreModelNCSNpp on bleed dataset")
    p.add_argument("--data_dir",       default="COMBINED/SLICED", help="Folder of sliced WAVs")
    p.add_argument("--checkpoint_dir", default="checkpoints",      help="Where to save models")
    p.add_argument("--batch_size",     type=int,   default=4,      help="Batch size")
    p.add_argument("--lr",             type=float, default=1e-3,   help="Learning rate")
    p.add_argument("--epochs",         type=int,   default=10,     help="Number of epochs")
    p.add_argument("--augment",        action="store_true",        help="Enable data augmentation")
    args, _ = p.parse_known_args()
    return args

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("▶ Device:",
