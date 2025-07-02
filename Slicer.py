import torchaudio
import os

# ─── CONFIG ─────────────────────────────────────
BT_DIR      = "COMBINED/BT"
VOX_DIR     = "COMBINED/VOX"
OUT_DIR     = "COMBINED/SLICED"
SLICE_SEC   = 10         # length of each slice, in seconds
SAMPLE_RATE = 44100      # your files’ sample rate
# ────────────────────────────────────────────────

def slice_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    bt_files = [f for f in os.listdir(BT_DIR) if f.endswith(" BT.wav")]
    print(f"Found {len(bt_files)} BT files to process.\n")

    for bt_file in sorted(bt_files):
        base_name = bt_file[:-len(" BT.wav")]   # strip off " BT.wav"
        vox_file  = f"{base_name} VOX.wav"

        bt_path  = os.path.join(BT_DIR, bt_file)
        vox_path = os.path.join(VOX_DIR, vox_file)

        if not os.path.exists(vox_path):
            print(f"⚠️ Skipping “{base_name}”: no VOX file found.")
            continue

        try:
            bt_wave, sr1  = torchaudio.load(bt_path)
            vox_wave, sr2 = torchaudio.load(vox_path)

            if sr1 != SAMPLE_RATE or sr2 != SAMPLE_RATE:
                print(f"⚠️ Skipping “{base_name}”: sample rate mismatch ({sr1} vs {sr2}).")
                continue

            slice_samples = SLICE_SEC * SAMPLE_RATE
            min_samples   = min(bt_wave.shape[1], vox_wave.shape[1])
            num_slices    = min_samples // slice_samples

            if num_slices < 1:
                dur = min_samples / SAMPLE_RATE
                print(f"⚠️ Skipping “{base_name}”: only {dur:.1f}s (< {SLICE_SEC}s).")
                continue

            for i in range(num_slices):
                start = i * slice_samples
                end   = start + slice_samples

                bt_slice  = bt_wave[:,  start:end]
                vox_slice = vox_wave[:, start:end]

                bt_out  = f"{base_name} BT_{i:03d}.wav"
                vox_out = f"{base_name} VOX_{i:03d}.wav"

                torchaudio.save(os.path.join(OUT_DIR, bt_out),  bt_slice,  SAMPLE_RATE)
                torchaudio.save(os.path.join(OUT_DIR, vox_out), vox_slice, SAMPLE_RATE)

            print(f"✅ {base_name}: {num_slices} × {SLICE_SEC}s slices saved.")

        except Exception as e:
            print(f"❌ Error processing “{base_name}”: {e}")

    print("\n🎉 All done — check", OUT_DIR)

if __name__ == "__main__":
    slice_all()