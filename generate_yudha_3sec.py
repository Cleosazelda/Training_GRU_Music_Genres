import pandas as pd

# Load features_3_sec.csv
df_3sec = pd.read_csv('d:/ITENAS/SMT6/DL/Implementasi_GRU/datasets/csv/Data/features_3_sec.csv')

# Drop columns that are not in datasetyudha.csv
# datasetyudha doesn't have: length, harmony, perceptr, tempo, and any _var columns
cols_to_drop = ['length', 'tempo']
cols_to_drop += [c for c in df_3sec.columns if '_var' in c or 'harmony' in c or 'perceptr' in c]

df_yudha_3sec = df_3sec.drop(columns=cols_to_drop)

# Rename _mean columns to match datasetyudha format
rename_map = {
    'chroma_stft_mean': 'chroma_stft',
    'rms_mean': 'rmse',
    'spectral_centroid_mean': 'spectral_centroid',
    'spectral_bandwidth_mean': 'spectral_bandwidth',
    'rolloff_mean': 'rolloff',
    'zero_crossing_rate_mean': 'zero_crossing_rate',
}
for i in range(1, 21):
    rename_map[f'mfcc{i}_mean'] = f'mfcc{i}'

df_yudha_3sec = df_yudha_3sec.rename(columns=rename_map)

# Reorder columns to match datasetyudha.csv exactly
expected_cols = [
    'filename', 'chroma_stft', 'rmse', 'spectral_centroid', 
    'spectral_bandwidth', 'rolloff', 'zero_crossing_rate'
] + [f'mfcc{i}' for i in range(1, 21)] + ['label']

df_yudha_3sec = df_yudha_3sec[expected_cols]

# Save the new dataset
save_path = 'd:/ITENAS/SMT6/DL/Implementasi_GRU/datasets/csv/datasetyudha_3_sec.csv'
df_yudha_3sec.to_csv(save_path, index=False)

print(f"Berhasil membuat dataset baru: {save_path}")
print(f"Total baris: {len(df_yudha_3sec)}")
print(f"Total kolom: {len(df_yudha_3sec.columns)}")
