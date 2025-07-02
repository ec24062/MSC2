import os
import random
import torch
torch.set_num_threads(1)
import torchaudio
from torch.utils.data import Dataset


def random_gain(wav: torch.Tensor, gain_db_range=(-3, 3)) -> torch.Tensor:
    """
    Apply a random gain shift in decibels to simulate level variation.
    wav: Tensor of shape [channels, samples]
    gain_db_range: tuple (min_db, max_db)
    """
    gain_db = random.uniform(*gain_db_range)
    factor = 10 ** (gain_db / 20)
    return wav * factor


def random_eq(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Apply a random peaking EQ boost or cut on one frequency band.
    """
    center_freq = random.uniform(200, 5000)
    Q = 0.707
    gain_db = random.uniform(-3, 3)
    effects = [[
        "equalizer",
        f"{center_freq}",
        f"{Q}",
        f"{gain_db}"
    ]]
    augmented, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sample_rate, effects)
    return augmented


def add_random_reverb(wav: torch.Tensor, sample_rate: int, prob: float = 0.3) -> torch.Tensor:
    """
    With probability `prob`, apply a mild reverb effect to simulate room acoustics.
    """
    if random.random() < prob:
        effects = [
            ["reverb", "50", "50", "100", "100", "0", "-w"]
        ]
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sample_rate, effects)
    return wav


class BleedDataset(Dataset):
    """
    PyTorch Dataset for paired backing-track bleed and vocal+bleed mixture slices.

    Expects a directory of WAV files named:
        "SONGNAME BT_xxx.wav" and "SONGNAME VOX_xxx.wav"
    where xxx is the zero-padded slice index.
    """
    def __init__(
        self,
        slices_dir: str,
        sample_rate: int = 44100,
        augment: bool = False
    ):
        self.slices_dir = slices_dir
        self.sample_rate = sample_rate
        self.augment = augment

        # Gather list of all slice files
        files = sorted(
            f for f in os.listdir(slices_dir)
            if f.lower().endswith('.wav')
        )
        # Partition into BT and VOX lists
        bt_files = [f for f in files if ' BT_' in f]
        vox_files = [f for f in files if ' VOX_' in f]

        # Build paired list: (bt_path, vox_path)
        self.pairs = []
        for bt in bt_files:
            base, idx = bt.rsplit(' BT_', 1)
            vox_name = f"{base} VOX_{idx}"
            if vox_name in vox_files:
                self.pairs.append((
                    os.path.join(slices_dir, bt),
                    os.path.join(slices_dir, vox_name)
                ))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        bt_path, vox_path = self.pairs[idx]
        bt_wave, _ = torchaudio.load(bt_path)
        vox_wave, _ = torchaudio.load(vox_path)

        if self.augment:
            # Apply independent augmentations
            bt_wave = random_gain(bt_wave)
            bt_wave = random_eq(bt_wave, self.sample_rate)
            bt_wave = add_random_reverb(bt_wave, self.sample_rate)

            vox_wave = random_gain(vox_wave)
            vox_wave = random_eq(vox_wave, self.sample_rate)
            vox_wave = add_random_reverb(vox_wave, self.sample_rate)

        return {
            'bleed_track': bt_wave,
            'mixture': vox_wave
        }
