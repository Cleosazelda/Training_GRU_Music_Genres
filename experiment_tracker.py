"""
experiment_tracker.py
─────────────────────
Modul untuk menyimpan, memuat, dan membandingkan seluruh hasil eksperimen
training secara persisten menggunakan local file storage (JSON + joblib).

Strategi Persistensi:
- registry.json  : Daftar meta-data semua eksperimen (akurasi, params, paths).
- models/<run_id>/: Direktori per eksperimen berisi model.keras, scaler, history.

Mengapa Local File Storage (bukan cloud DB)?
- Tidak membutuhkan koneksi internet / konfigurasi tambahan.
- Cocok untuk proyek akademis / prototyping.
- joblib efisien untuk numpy arrays (scaler, history); JSON untuk metadata ringan.
- Dapat di-upgrade ke cloud (GCS, S3) hanya dengan mengganti fungsi save/load.
"""

import json
import os
import shutil
from datetime import datetime
import numpy as np
import joblib
import pandas as pd

REGISTRY_PATH = os.path.join("models", "registry.json")
MODELS_DIR    = "models"


# ─── Registry I/O ─────────────────────────────────────────────────────────────

def _load_registry() -> list:
    """Memuat seluruh rekaman eksperimen dari registry.json."""
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_registry(registry: list) -> None:
    """Menulis ulang registry.json dengan data terbaru."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ─── Experiment I/O ───────────────────────────────────────────────────────────

def generate_run_id(dataset_name: str) -> str:
    """
    Menghasilkan ID unik untuk satu sesi training.
    Format: run_YYYYMMDD_HHMMSS_<dataset_slug>
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = dataset_name.replace(" ", "_").replace("/", "-")[:20]
    return f"run_{ts}_{slug}"


def save_experiment(run_id: str, dataset_name: str, hyperparams: dict,
                    metrics: dict, history: dict,
                    model_obj, scaler_obj, encoder_obj, classes_arr) -> str:
    """
    Menyimpan satu eksperimen secara lengkap ke disk.

    Struktur direktori yang dibuat:
        models/<run_id>/
            model.keras      ← bobot DNN terbaik
            history.pkl      ← riwayat loss/accuracy per epoch
            scaler.pkl       ← StandardScaler yang di-fit pada train data
            label_encoder.pkl← LabelEncoder untuk nama kelas
            classes.pkl      ← array nama kelas

    Meta-data (ringan) ditambahkan ke registry.json agar dapat dibaca
    tanpa memuat seluruh model ke memori.

    Args:
        run_id (str)       : ID unik dari generate_run_id().
        dataset_name (str) : Nama dataset yang digunakan.
        hyperparams (dict) : Konfigurasi hyperparameter model.
        metrics (dict)     : {'accuracy', 'precision', 'recall', 'f1',
                               'val_accuracy', 'val_loss', 'epochs_run'}.
        history (dict)     : history.history dari Keras.
        model_obj          : tf.keras.Model yang sudah ditraining.
        scaler_obj         : Fitted StandardScaler.
        encoder_obj        : Fitted LabelEncoder.
        classes_arr        : numpy array nama kelas.

    Returns:
        str: Path direktori eksperimen yang baru dibuat.
    """
    run_dir = os.path.join(MODELS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Simpan artefak berat
    model_path = os.path.join(run_dir, "model.keras")
    model_obj.save(model_path)
    joblib.dump(history,      os.path.join(run_dir, "history.pkl"))
    joblib.dump(scaler_obj,   os.path.join(run_dir, "scaler.pkl"))
    joblib.dump(encoder_obj,  os.path.join(run_dir, "label_encoder.pkl"))
    joblib.dump(classes_arr,  os.path.join(run_dir, "classes.pkl"))

    # Bangun entri metadata (hanya tipe primitif agar JSON-serializable)
    def _to_serializable(v):
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.bool_,)):    return bool(v)
        return v

    clean_metrics    = {k: _to_serializable(v) for k, v in metrics.items()}
    clean_hyperparams = {k: _to_serializable(v) for k, v in hyperparams.items()}

    entry = {
        "run_id":       run_id,
        "timestamp":    datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "hyperparams":  clean_hyperparams,
        "metrics":      clean_metrics,
        "run_dir":      run_dir,
        "model_path":   model_path,
    }

    registry = _load_registry()
    # Hapus entri lama dengan run_id yang sama (jika ada retrain)
    registry = [r for r in registry if r["run_id"] != run_id]
    registry.append(entry)
    _save_registry(registry)

    return run_dir


def load_all_experiments() -> list:
    """
    Memuat seluruh rekaman eksperimen dari registry.
    Memfilter entri yang direktori modelnya sudah tidak ada di disk.

    Returns:
        list[dict]: Daftar entri eksperimen yang valid.
    """
    registry = _load_registry()
    valid = [r for r in registry if os.path.exists(r.get("run_dir", ""))]
    return valid


def delete_experiment(run_id: str) -> bool:
    """
    Menghapus satu eksperimen: direktori artefak + entri di registry.

    Args:
        run_id (str): ID eksperimen yang akan dihapus.

    Returns:
        bool: True jika berhasil.
    """
    registry = _load_registry()
    entry = next((r for r in registry if r["run_id"] == run_id), None)
    if entry and os.path.exists(entry["run_dir"]):
        shutil.rmtree(entry["run_dir"])
    registry = [r for r in registry if r["run_id"] != run_id]
    _save_registry(registry)
    return True


# ─── Leaderboard Utilities ────────────────────────────────────────────────────

def build_leaderboard_df(experiments: list) -> pd.DataFrame:
    """
    Membangun DataFrame leaderboard dari daftar eksperimen.

    Kolom output:
        Run ID | Timestamp | Dataset | Hidden Layers | Units | Aktivasi |
        Optimizer | LR | Dropout | Val Acc | Acc | Precision | Recall | F1 | Epochs

    Returns:
        pd.DataFrame: Leaderboard diurutkan berdasarkan Val Accuracy (DESC).
    """
    rows = []
    for exp in experiments:
        m  = exp.get("metrics", {})
        hp = exp.get("hyperparams", {})
        rows.append({
            "Run ID":         exp["run_id"],
            "Timestamp":      exp["timestamp"][:19].replace("T", " "),
            "Dataset":        exp.get("dataset_name", "-"),
            "Layers":         hp.get("hidden_layers", "-"),
            "Units":          hp.get("hidden_units", "-"),
            "Aktivasi":       hp.get("dense_activation", "-"),
            "Optimizer":      hp.get("optimizer_name", "-"),
            "LR":             hp.get("learning_rate", "-"),
            "Dropout":        hp.get("dropout_rate", "-"),
            "Val Acc (%)":    round(m.get("val_accuracy", 0) * 100, 2),
            "Train Acc (%)":  round(m.get("train_accuracy", 0) * 100, 2),
            "Precision (%)":  round(m.get("precision", 0) * 100, 2),
            "Recall (%)":     round(m.get("recall", 0) * 100, 2),
            "F1-Score (%)":   round(m.get("f1", 0) * 100, 2),
            "Epochs":         m.get("epochs_run", "-"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Val Acc (%)", ascending=False).reset_index(drop=True)
    df.index += 1  # Rank mulai dari 1
    return df


def get_best_model(experiments: list) -> dict | None:
    """
    Menentukan model terbaik (Overall Champion) berdasarkan Val Accuracy tertinggi.
    Tie-breaker: F1-Score tertinggi.

    Args:
        experiments (list): Output dari load_all_experiments().

    Returns:
        dict | None: Entri eksperimen terbaik, atau None jika kosong.
    """
    if not experiments:
        return None
    return max(
        experiments,
        key=lambda e: (
            e.get("metrics", {}).get("val_accuracy", 0),
            e.get("metrics", {}).get("f1", 0)
        )
    )


def get_best_per_dataset(experiments: list) -> dict:
    """
    Menentukan model terbaik per dataset.

    Returns:
        dict: {dataset_name: best_experiment_entry}
    """
    best = {}
    for exp in experiments:
        ds  = exp.get("dataset_name", "Unknown")
        acc = exp.get("metrics", {}).get("val_accuracy", 0)
        if ds not in best or acc > best[ds]["metrics"].get("val_accuracy", 0):
            best[ds] = exp
    return best


# ─── Dataset Upload Validation ────────────────────────────────────────────────

REQUIRED_COLS = {
    'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var',
    'spectral_bandwidth_mean', 'spectral_bandwidth_var',
    'rolloff_mean', 'rolloff_var',
    'zero_crossing_rate_mean', 'zero_crossing_rate_var',
    'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var',
    'tempo', 'label'
}

def validate_uploaded_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Memvalidasi format DataFrame CSV yang diunggah pengguna.

    Checks:
        1. Kolom 'label' ada.
        2. Minimal fitur inti (MFCC, chroma, tempo, dll) tersedia.
        3. Tidak ada baris yang seluruhnya kosong.
        4. Kolom fitur numerik, bukan string.

    Args:
        df (pd.DataFrame): DataFrame hasil pd.read_csv().

    Returns:
        tuple[bool, str]: (is_valid, pesan_error_atau_sukses)
    """
    if 'label' not in df.columns:
        return False, "❌ Kolom 'label' tidak ditemukan. Dataset harus memiliki kolom 'label'."

    missing_core = REQUIRED_COLS - set(df.columns)
    # Cek minimal: harus ada 'label' dan setidaknya kolom MFCC
    mfcc_cols = [c for c in df.columns if 'mfcc' in c.lower()]
    if len(mfcc_cols) < 2:
        return False, (f"❌ Kolom fitur MFCC tidak ditemukan. "
                       f"Pastikan dataset memiliki kolom seperti 'mfcc1_mean', 'mfcc1_var', dll.")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in df.columns if c not in ('label', 'filename', 'length')]
    non_numeric = [c for c in feature_cols if c not in num_cols]
    if non_numeric:
        return False, f"❌ Kolom berikut tidak numerik: {non_numeric[:5]}. Semua fitur harus berupa angka."

    if df.dropna(how='all').empty:
        return False, "❌ Dataset kosong setelah menghapus baris NaN."

    num_classes = df['label'].nunique()
    if num_classes < 2:
        return False, f"❌ Dataset hanya memiliki {num_classes} kelas. Minimal 2 kelas dibutuhkan."

    null_pct = df.isnull().mean().max() * 100
    warning = ""
    if null_pct > 5:
        warning = f" ⚠️ Peringatan: {null_pct:.1f}% nilai kosong terdeteksi dan akan dihapus otomatis."

    return True, (f"✅ Dataset valid: {len(df):,} baris | {len(feature_cols)} fitur | "
                  f"{num_classes} kelas genre.{warning}")


def save_uploaded_dataset(df: pd.DataFrame, filename: str) -> str:
    """
    Menyimpan DataFrame yang telah divalidasi ke direktori datasets lokal.

    Args:
        df (pd.DataFrame): DataFrame hasil upload.
        filename (str): Nama file asal upload.

    Returns:
        str: Path file CSV yang tersimpan.
    """
    upload_dir = os.path.join("FMA_like_GTZAN_for_Music_Genre_Classification", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    save_path = os.path.join(upload_dir, f"{ts}_{safe_name}")
    df.to_csv(save_path, index=False)
    return save_path
