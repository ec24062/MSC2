# train_diffsep.py

import os
import argparse
import torch
import importlib.util
from torch.utils.data import DataLoader
from bleed_dataset import BleedDataset

def find_and_import_diffsep(repo_dir: str):
    """
    Walks repo_dir and finds a .py file that defines class DiffSep,
    then dynamically imports it and returns the module.
    """
    for root, _, files in os.walk(repo_dir):
        for fn in files:
            if fn.lower().endswith(".py"):
                path = os.path.join(root, fn)
                # Quick check: scan the file for "class DiffSep"
                try:
                    with open(path, "r") as f:
                        text = f.read()
                    if "class DiffSep" in text:
                        spec = importlib.util.spec_from_file_location("diffsep_local", path)
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        return mod
                except Exception:
                    continue
    raise FileNotFoundError(f"No file defining 'class DiffSep' found under {repo_dir}")

def parse_args():
    p = argparse.ArgumentParser(description="Train DiffSep on bleed dataset")
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

    # 1) Find and import your local DiffSep implementation
    repo = "diffusion-separation-main"
    print(f"🔍 Searching for DiffSep under '{repo}'...")
    diffmod = find_and_import_diffsep(repo)
    print(f"✅ Loaded DiffSep from: {diffmod.__file__}")
    model = diffmod.DiffSep().to(device)

    # 2) Data loader
    ds     = BleedDataset(args.data_dir, sample_rate=44100, augment=args.augment)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    # 3) Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 4) Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            mix   = batch["mixture"].to(device)      # [B, C, T]
            bleed = batch["bleed_track"].to(device)  # [B, C, T]

            optimizer.zero_grad()
            # DiffSep’s forward almost always just takes the mix:
            est = model(mix)
            # If it returns a dict, pick the first tensor
            if isinstance(est, dict):
                est = list(est.values())[0]

            loss = criterion(est, bleed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mix.size(0)

        avg = total_loss / len(ds)
        print(f"Epoch {epoch}/{args.epochs} — Loss: {avg:.4f}")

        ckpt = os.path.join(args.checkpoint_dir, f"diffsep_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt)

    print("✅ Training complete. Checkpoints saved to", args.checkpoint_dir)

if __name__ == "__main__":
    main()