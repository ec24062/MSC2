import os
import argparse
import torch
import importlib.util
from torch.utils.data import DataLoader
from bleed_dataset import BleedDataset

def import_score_model(repo_dir: str):
    """
    Directly imports ScoreModelNCSNpp from score_models.py in the repo_dir/models folder.
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
    print("▶ Device:", device)

    # Import ScoreModelNCSNpp
    repo = "diffusion-separation-main"
    print(f"🔍 Importing ScoreModelNCSNpp from '{repo}/models/score_models.py'...")
    ScoreModel = import_score_model(repo)
    print(f"✅ Loaded ScoreModelNCSNpp")

    # Create model (you may need to adjust these arguments depending on your config)
    stft_args = dict(n_fft=1024, hop_length=256)
    backbone_args = dict(
        _target_="models.cdiffuse_network.DiffuSE",
        sigma_max=0.5,
        sigma_min=0.01,
        num_scales=1000
    )
    model = ScoreModel(
        num_sources=1,
        stft_args=stft_args,
        backbone_args=backbone_args
    ).to(device)

    # Dataset & loader
    ds     = BleedDataset(args.data_dir, sample_rate=44100, augment=args.augment)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            mix   = batch["mixture"].to(device)      # [B, C, T]
            bleed = batch["bleed_track"].to(device)  # [B, C, T]

            optimizer.zero_grad()
            est = model(mix, None, mix)  # Add time_cond argument if needed
            if isinstance(est, dict):
                est = list(est.values())[0]

            loss = criterion(est, bleed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mix.size(0)

        avg = total_loss / len(ds)
        print(f"Epoch {epoch}/{args.epochs} — Loss: {avg:.4f}")

        ckpt = os.path.join(args.checkpoint_dir, f"scoremodel_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt)

    print("✅ Training complete. Checkpoints saved to", args.checkpoint_dir)

if __name__ == "__main__":
    main()
