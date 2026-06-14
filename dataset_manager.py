# ─── Dataset Library Manager ──────────────────────────────────────────────────
"""
Modul untuk manajemen dataset dinamis.

Struktur folder dataset:
  datasets/
  ├── csv/          ← Taruh file CSV di sini (auto-detected)
  │   ├── fma_3secs.csv
  │   ├── fma_30secs.csv
  │   └── (dataset lain...)
  └── audio/        ← Untuk dataset audio mentah (GTZAN, dll)
      └── GTZAN/
          └── genres_original/
              ├── blues/
              ├── classical/
              └── ...
"""

import os
import pandas as pd
import numpy as np


# ─── Folder Constants ──────────────────────────────────────────────────────────
CSV_DATASET_DIR   = "datasets/csv"
AUDIO_DATASET_DIR = "datasets/audio"
LEGACY_DIR        = "FMA_like_GTZAN_for_Music_Genre_Classification"

os.makedirs(CSV_DATASET_DIR, exist_ok=True)
os.makedirs(AUDIO_DATASET_DIR, exist_ok=True)


def discover_csv_datasets() -> dict:
    """
    Scan semua file CSV di folder datasets/csv/ dan folder legacy FMA.
    Returns dict: {display_name: filepath}
    """
    datasets = {}

    # 1. Scan folder baru datasets/csv/
    if os.path.isdir(CSV_DATASET_DIR):
        for fname in sorted(os.listdir(CSV_DATASET_DIR)):
            if fname.lower().endswith('.csv'):
                fpath = os.path.join(CSV_DATASET_DIR, fname)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                display = f"{os.path.splitext(fname)[0]} ({size_mb:.1f} MB)"
                datasets[display] = fpath

    # 2. Scan folder legacy (FMA_like_GTZAN) untuk backward compatibility
    if os.path.isdir(LEGACY_DIR):
        for fname in sorted(os.listdir(LEGACY_DIR)):
            if fname.lower().endswith('.csv'):
                fpath = os.path.join(LEGACY_DIR, fname)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                display = f"[Legacy] {os.path.splitext(fname)[0]} ({size_mb:.1f} MB)"
                datasets[display] = fpath

    return datasets


def get_dataset_info(filepath: str) -> dict:
    """
    Baca informasi dasar sebuah dataset CSV tanpa memuat seluruhnya.
    Returns: dict berisi num_rows, num_cols, num_classes, classes
    """
    try:
        df = pd.read_csv(filepath, nrows=5000)
        full_count = sum(1 for _ in open(filepath, encoding='utf-8')) - 1
        info = {
            "filepath":    filepath,
            "filename":    os.path.basename(filepath),
            "size_mb":     round(os.path.getsize(filepath) / (1024**2), 2),
            "num_rows":    full_count,
            "num_cols":    len(df.columns),
            "num_classes": df['label'].nunique() if 'label' in df.columns else '?',
            "classes":     sorted(df['label'].unique().tolist()) if 'label' in df.columns else [],
            "has_label":   'label' in df.columns,
            "sample_cols": list(df.columns[:8]),
        }
        return info
    except Exception as e:
        return {"filepath": filepath, "error": str(e)}


def save_csv_to_library(df: pd.DataFrame, filename: str) -> str:
    """
    Simpan DataFrame ke folder datasets/csv/ dengan nama yang aman.
    Returns: path lengkap file yang disimpan.
    """
    # Bersihkan nama file
    safe_name = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
    if not safe_name.lower().endswith('.csv'):
        safe_name += '.csv'

    save_path = os.path.join(CSV_DATASET_DIR, safe_name)

    # Jika sudah ada, tambahkan suffix
    base, ext = os.path.splitext(save_path)
    counter = 1
    while os.path.exists(save_path):
        save_path = f"{base}_{counter}{ext}"
        counter += 1

    df.to_csv(save_path, index=False)
    return save_path


def validate_csv_for_training(df: pd.DataFrame) -> tuple:
    """
    Validasi apakah CSV kompatibel untuk training GRU/DNN.
    Returns: (is_valid: bool, message: str)
    """
    if 'label' not in df.columns:
        return False, "❌ Kolom 'label' tidak ditemukan. Dataset harus memiliki kolom genre label."

    feature_cols = [c for c in df.columns if c not in ('label', 'filename', 'Unnamed: 0')]
    if len(feature_cols) < 5:
        return False, f"❌ Hanya {len(feature_cols)} kolom fitur ditemukan. Minimal 5 kolom fitur diperlukan."

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        return False, f"❌ Kolom non-numerik ditemukan: {non_numeric[:5]}. Semua fitur harus berupa angka."

    n_classes = df['label'].nunique()
    if n_classes < 2:
        return False, f"❌ Hanya {n_classes} kelas label. Dibutuhkan minimal 2 kelas berbeda."

    null_pct = df[feature_cols].isnull().mean().mean() * 100
    warn = ""
    if null_pct > 5:
        warn = f" ⚠️ {null_pct:.1f}% nilai kosong — akan dihapus otomatis saat training."

    return True, (
        f"✅ Dataset valid: {len(df):,} baris | {len(feature_cols)} fitur | "
        f"{n_classes} kelas genre.{warn}"
    )


def get_audio_datasets() -> dict:
    """
    Scan folder datasets/audio/ untuk dataset audio mentah (GTZAN, dll).
    Returns dict: {dataset_name: base_path}
    """
    audio_ds = {}
    if os.path.isdir(AUDIO_DATASET_DIR):
        for dname in sorted(os.listdir(AUDIO_DATASET_DIR)):
            dpath = os.path.join(AUDIO_DATASET_DIR, dname)
            if os.path.isdir(dpath):
                # Hitung jumlah file audio
                n_audio = sum(
                    1 for _, _, files in os.walk(dpath)
                    for f in files if f.lower().endswith(('.wav', '.mp3', '.ogg'))
                )
                audio_ds[f"{dname} ({n_audio} audio files)"] = dpath
    return audio_ds
