import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_and_preprocess_data(csv_path):
    """Loads and preprocesses the GTZAN-like CSV dataset."""
    df = pd.read_csv(csv_path)
    
    # Drop filename
    if 'filename' in df.columns:
        df = df.drop(columns=['filename'])
    
    # Separate features and labels
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and encoder for inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    # Reshape for GRU: (batch_size, timesteps, features)
    # Here, we treat the 57 features as 57 timesteps of 1 feature, or 1 timestep of 57 features.
    # Standard 1D sequence for GRU is (batch_size, sequence_length, 1)
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    return X_train_reshaped, X_test_reshaped, y_train, y_test, label_encoder.classes_

def extract_features_from_audio(file_path):
    """Extracts exactly 57 features from an audio file to match the dataset."""
    y, sr = librosa.load(file_path, duration=30) # load up to 30 seconds
    
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
    
    # 13-16. Harmony and Perceptrual
    harmony, perceptr = librosa.effects.hpss(y)
    features.extend([np.mean(harmony), np.var(harmony)])
    features.extend([np.mean(perceptr), np.var(perceptr)])
    
    # 17. Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
    
    # 18-57. MFCC 1 to 20
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features.extend([np.mean(mfcc[i]), np.var(mfcc[i])])
        
    return np.array(features)

def preprocess_audio_for_inference(file_path):
    """Extracts features and applies saved scaler and reshaping for GRU."""
    features = extract_features_from_audio(file_path)
    features = features.reshape(1, -1) # (1, 57)
    
    try:
        scaler = joblib.load('models/scaler.pkl')
        features_scaled = scaler.transform(features)
    except FileNotFoundError:
        # Fallback if no scaler is found (e.g. model not trained yet)
        features_scaled = features
        
    # Reshape for GRU: (1, 57, 1)
    features_reshaped = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
    return features_reshaped
