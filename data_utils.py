import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


def load_and_preprocess_data(csv_path):
    """
    Memuat dan memproses dataset CSV fitur statistik audio (FMA-like / GTZAN-like).

    Pipeline Normalisasi (StandardScaler):
    ----------------------------------------
    StandardScaler mengubah setiap fitur sehingga memiliki mean=0 dan std=1.
    Ini sangat penting untuk data tabular dengan skala yang sangat berbeda:
    - 'tempo' bisa bernilai 60-200 BPM
    - 'mfcc_var' bisa bernilai 0.001 - 5000
    Tanpa normalisasi, fitur berskala besar akan mendominasi gradien saat
    backpropagation, menyebabkan model sulit konvergen dan akurasi rendah.

    Scaler difit HANYA pada data training (X_train) untuk menghindari
    data leakage, lalu diaplikasikan ke X_test.

    Args:
        csv_path (str): Path ke file CSV dataset.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, class_names)
               X sudah dinormalisasi dan dalam format np.float32 (efisien untuk GPU).
    """
    df = pd.read_csv(csv_path)

    # Hapus kolom non-fitur
    cols_to_drop = [c for c in ['filename', 'length'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Hapus baris yang mengandung NaN/Inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Pisahkan fitur (X) dan label (y)
    X = df.drop(columns=['label']).values
    y = df['label'].values

    # Encode label string → integer
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data: 80% train, 20% test, stratified agar distribusi kelas seimbang
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ---- Normalisasi dengan StandardScaler ----
    # fit() hanya pada X_train → transform() pada keduanya
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)

    # Simpan scaler & encoder untuk digunakan saat inferensi audio
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')

    return X_train_scaled, X_test_scaled, y_train.astype(np.int32), y_test.astype(np.int32), label_encoder.classes_


def extract_features_from_audio(file_path):
    """
    Mengekstrak 57 fitur statistik dari file audio agar cocok dengan format dataset CSV.

    Fitur yang diekstrak:
        - Chroma STFT (mean, var)
        - RMS (mean, var)
        - Spectral Centroid (mean, var)
        - Spectral Bandwidth (mean, var)
        - Rolloff (mean, var)
        - Zero Crossing Rate (mean, var)
        - Harmony (mean, var)
        - Perceptr (mean, var)
        - Tempo (1 nilai)
        - MFCC 1-20 (mean dan var masing-masing → 40 nilai)
        Total: 17 + 40 = 57 fitur

    Args:
        file_path (str): Path ke file audio (.wav, .mp3, dll).

    Returns:
        np.ndarray: Array 1D dengan 57 nilai fitur statistik.
    """
    y, sr = librosa.load(file_path, duration=30)

    features = []

    # 1-2. Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend([np.mean(chroma_stft), np.var(chroma_stft)])

    # 3-4. RMS
    rms = librosa.feature.rms(y=y)
    features.extend([np.mean(rms), np.var(rms)])

    # 5-6. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend([np.mean(spec_cent), np.var(spec_cent)])

    # 7-8. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.extend([np.mean(spec_bw), np.var(spec_bw)])

    # 9-10. Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend([np.mean(rolloff), np.var(rolloff)])

    # 11-12. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend([np.mean(zcr), np.var(zcr)])

    # 13-16. Harmony & Perceptr
    harmony, perceptr = librosa.effects.hpss(y)
    features.extend([np.mean(harmony), np.var(harmony)])
    features.extend([np.mean(perceptr), np.var(perceptr)])

    # 17. Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo))

    # 18-57. MFCC 1-20 (mean + var)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features.extend([np.mean(mfcc[i]), np.var(mfcc[i])])

    return np.array(features, dtype=np.float32)


def preprocess_audio_for_inference(file_path):
    """
    Pipeline inferensi audio: ekstrak fitur → normalisasi dengan scaler tersimpan.

    Tidak ada reshape seperti GRU karena model Dense menerima input 1D (flat vector).

    Args:
        file_path (str): Path ke file audio.

    Returns:
        np.ndarray: Array shape (1, 57) siap untuk model.predict().
    """
    features = extract_features_from_audio(file_path)
    features = features.reshape(1, -1)  # shape: (1, 57)

    try:
        scaler = joblib.load('models/scaler.pkl')
        features_scaled = scaler.transform(features).astype(np.float32)
    except FileNotFoundError:
        print("[WARNING] scaler.pkl tidak ditemukan, fitur tidak dinormalisasi.")
        features_scaled = features

    return features_scaled  # shape: (1, 57) — langsung kompatibel dengan Dense layer
