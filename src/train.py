# src/train.py
import os
import numpy as np
import joblib

# =========================
# PATH
# =========================
DATA_DIR = "data/prototypes"
OUT_PATH = "data/models/gesture_model.joblib"

# =========================
# PARAMETER
# =========================
MAX_SEQ_LEN = 120   # normalisasi panjang sequence (aman untuk dinamis)
MIN_SEQ_LEN = 10    # buang data rusak / terlalu pendek

# =========================
# UTIL
# =========================
def normalize_sequence(seq, max_len=MAX_SEQ_LEN):
    """
    Samakan panjang sequence agar DTW lebih stabil
    (downsample jika terlalu panjang)
    """
    T = len(seq)
    if T <= max_len:
        return seq

    idx = np.linspace(0, T - 1, max_len).astype(int)
    return seq[idx]


def load_dataset():
    """
    Load semua prototype gesture dari folder
    Struktur:
    data/prototypes/
      ├─ A/
      ├─ B/
      ├─ ME/
      ├─ SAYA/
      └─ BUDI/
    """
    dataset = {}
    total_samples = 0

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {DATA_DIR}")

    for label in sorted(os.listdir(DATA_DIR)):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        samples = []
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".npy"):
                continue

            path = os.path.join(label_dir, fname)
            seq = np.load(path)

            # Validasi bentuk data
            if len(seq) < MIN_SEQ_LEN:
                print(f"⚠️  Skip {fname} (terlalu pendek)")
                continue

            seq = normalize_sequence(seq)
            samples.append(seq)

        if samples:
            dataset[label] = samples
            total_samples += len(samples)
            print(f"✓ {label}: {len(samples)} samples")

    if total_samples == 0:
        raise RuntimeError("Dataset kosong atau tidak valid")

    return dataset, total_samples


# =========================
# MAIN TRAIN
# =========================
def main():
    print("=== TRAIN GESTURE MODEL (DTW PROTOTYPE) ===\n")

    dataset, total = load_dataset()

    labels = sorted(dataset.keys())
    min_per_class = min(len(v) for v in dataset.values())
    max_per_class = max(len(v) for v in dataset.values())

    print("\n=== DATASET SUMMARY ===")
    print(f"Total label        : {len(labels)}")
    print(f"Total sample       : {total}")
    print(f"Min sample / label : {min_per_class}")
    print(f"Max sample / label : {max_per_class}")

    model = {
        "type": "DTW_PROTOTYPE",
        "labels": labels,
        "dataset": dataset,
        "config": {
            "max_seq_len": MAX_SEQ_LEN,
            "min_seq_len": MIN_SEQ_LEN
        }
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump(model, OUT_PATH)

    print(f"\n✅ Model berhasil disimpan:")
    print(f"   {OUT_PATH}")


if __name__ == "__main__":
    main()
