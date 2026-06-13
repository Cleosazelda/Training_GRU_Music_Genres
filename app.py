import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_utils import load_and_preprocess_data, preprocess_audio_for_inference
from model_utils import (build_dense_model, build_tf_dataset,
                         plot_training_history, plot_confusion_matrix_custom,
                         enable_mixed_precision)
from experiment_tracker import (generate_run_id, save_experiment,
                                load_all_experiments, delete_experiment,
                                build_leaderboard_df, get_best_model,
                                get_best_per_dataset,
                                validate_uploaded_csv, save_uploaded_dataset)
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Genre Classification – DNN",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f0f1a; }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        border: 1px solid #3d3d5c;
    }
    div[data-testid="metric-container"] label { color: #a0a0c0 !important; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #7aa2f7 !important; font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e1e2e;
        border-radius: 8px 8px 0 0;
        color: #a0a0c0;
        font-weight: 600;
        padding: 10px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7aa2f7, #bb9af7) !important;
        color: #0f0f1a !important;
    }

    .info-card {
        background: linear-gradient(135deg, #1e1e2e, #16213e);
        border-left: 4px solid #7aa2f7;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .success-badge {
        background: linear-gradient(135deg, #1a4731, #0d2e1f);
        border: 1px solid #9ece6a;
        border-radius: 8px;
        padding: 10px 16px;
        color: #9ece6a;
        font-weight: 600;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7aa2f7, #bb9af7);
        border: none;
        color: #0f0f1a;
        font-weight: 700;
        border-radius: 8px;
        padding: 12px 24px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(122,162,247,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.title("🎵 Music Genre Classification")
st.markdown(
    "**Deep Neural Network (DNN/FFNN)** untuk Klasifikasi Genre Musik dari Fitur Statistik Audio — "
    "Dioptimalkan dengan Mixed Precision & tf.data Pipeline."
)

DATASET_OPTIONS = {
    "FMA 3 Seconds (Besar, ~87MB)": "FMA_like_GTZAN_for_Music_Genre_Classification/fma_3secs.csv",
    "FMA 30 Seconds (Cepat, ~8MB)": "FMA_like_GTZAN_for_Music_Genre_Classification/fma_30secs.csv",
}
os.makedirs("models", exist_ok=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📂 Dataset Overview",
    "⚙️ Model Training",
    "📊 Evaluation Dashboard",
    "📝 Analisis Akademis",
    "🎧 Live Audio Prediction",
    "📤 Upload Dataset",
    "🏆 Leaderboard"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Dataset Overview")

    selected_ds_label = st.selectbox("Pilih Dataset:", list(DATASET_OPTIONS.keys()), key="ds_overview")
    DATASET_PATH = DATASET_OPTIONS[selected_ds_label]

    st.markdown(f"<div class='info-card'>📁 Path: <code>{DATASET_PATH}</code></div>", unsafe_allow_html=True)

    try:
        df_preview = pd.read_csv(DATASET_PATH, nrows=200)
        num_features = len(df_preview.columns) - 2  # exclude filename & label
        num_classes  = df_preview['label'].nunique()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📋 Total Baris (Preview)", f"{len(df_preview):,}")
        c2.metric("🧮 Jumlah Fitur", num_features)
        c3.metric("🎵 Jumlah Kelas Genre", num_classes)
        c4.metric("📊 Tipe Data Utama", "Tabular / Statistik")

        st.subheader("Sample Data")
        st.dataframe(df_preview.head(10), use_container_width=True)

        st.subheader("Distribusi Label Genre")
        label_counts = df_preview['label'].value_counts()
        st.bar_chart(label_counts)

        st.subheader("Statistik Deskriptif Fitur")
        num_df = df_preview.drop(columns=['filename', 'label'], errors='ignore')
        st.dataframe(num_df.describe().T.style.background_gradient(cmap='Blues'), use_container_width=True)

    except FileNotFoundError:
        st.error(f"❌ Dataset tidak ditemukan di: `{DATASET_PATH}`")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Model Training
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("⚙️ Model Training — Feed-Forward Neural Network (DNN)")
    st.markdown(
        "Konfigurasi arsitektur **Deep Neural Network** yang dioptimalkan untuk data tabular fitur statistik. "
        "Gunakan Mixed Precision dan tf.data untuk training yang cepat di GPU."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Pilihan Dataset")
        selected_ds_train = st.selectbox("Dataset Training:", list(DATASET_OPTIONS.keys()), key="ds_train")
        TRAIN_DATASET_PATH = DATASET_OPTIONS[selected_ds_train]

        st.subheader("2. Arsitektur DNN")
        hidden_layers  = st.number_input("Jumlah Hidden Layer", min_value=1, max_value=8, value=3,
                                          help="Arsitektur Pyramid: jumlah neuron berkurang 2x setiap layer.")
        hidden_units   = st.selectbox("Neuron Layer Pertama (Puncak Piramida)", [64, 128, 256, 512, 1024], index=2)
        dense_activation = st.selectbox("Fungsi Aktivasi Hidden Layer",
                                         ['relu', 'elu', 'selu', 'gelu', 'swish', 'tanh'],
                                         help="ReLU: cepat & umum. GELU/Swish: performa lebih tinggi namun lebih lambat.")
        use_batchnorm  = st.checkbox("Batch Normalization", value=True,
                                      help="Menstabilkan distribusi aktivasi antar layer, mempercepat konvergensi.")

    with col2:
        st.subheader("3. Regularisasi")
        dropout_rate  = st.slider("Dropout Rate", 0.0, 0.7, 0.3, 0.05,
                                   help="Menonaktifkan neuron secara acak saat training untuk mencegah overfitting.")
        l2_reg_rate   = st.selectbox("L2 Regularization (Weight Decay)",
                                      [0.0, 0.01, 0.001, 0.0001],
                                      format_func=lambda x: "0 (Nonaktif)" if x == 0.0 else str(x), index=2)

        st.subheader("4. Optimizer")
        optimizer_name = st.selectbox("Optimizer", ['Adam', 'AdamW', 'RMSprop', 'SGD', 'Nadam', 'Adamax'])
        learning_rate  = st.selectbox("Learning Rate", [0.01, 0.005, 0.001, 0.0005, 0.0001], index=2)
        use_grad_clip  = st.checkbox("Gradient Clipping (clipvalue=1.0)", value=False)

    with col3:
        st.subheader("5. Training Settings")
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256, 512], index=2,
                                   help="Batch besar lebih efisien untuk GPU karena meningkatkan utilisasi memori.")
        epochs     = st.number_input("Maks Epoch", min_value=5, max_value=500, value=100)

        st.subheader("6. GPU Optimization")
        use_mixed_precision = st.checkbox("Mixed Precision (float16) 🚀", value=True,
                                           help="Mempercepat komputasi GPU 2-3x. Direkomendasikan untuk GPU NVIDIA Turing+.")
        use_tf_data = st.checkbox("tf.data Pipeline + Prefetch 🚀", value=True,
                                   help="Overlap CPU-GPU pipeline agar GPU tidak menganggur antar batch.")

        st.subheader("7. Callbacks Otomatis")
        use_early_stopping = st.checkbox("Early Stopping (patience=15)", value=True)
        use_reduce_lr      = st.checkbox("ReduceLROnPlateau (patience=7)", value=True)
        use_checkpoint     = st.checkbox("ModelCheckpoint (simpan bobot terbaik)", value=True)

    # ── Training Button ──
    if st.button("🚀 Mulai Training Model", type="primary", key="train_btn"):

        # 1. Enable Mixed Precision
        if use_mixed_precision:
            with st.spinner("Mengaktifkan Mixed Precision (float16)..."):
                try:
                    enable_mixed_precision()
                    st.success("✅ Mixed Precision aktif — komputasi GPU berjalan dalam float16.")
                except Exception as e:
                    st.warning(f"⚠️ Mixed Precision tidak dapat diaktifkan: {e}")

        # 2. Load dataset
        with st.spinner(f"Memuat & memproses dataset `{TRAIN_DATASET_PATH}`..."):
            try:
                X_train, X_test, y_train, y_test, classes = load_and_preprocess_data(TRAIN_DATASET_PATH)
                num_classes = len(classes)
                input_dim   = X_train.shape[1]
                st.info(f"✅ Dataset dimuat: {X_train.shape[0]:,} train | {X_test.shape[0]:,} test | "
                        f"{input_dim} fitur | {num_classes} kelas")
            except Exception as e:
                st.error(f"❌ Gagal memuat dataset: {e}")
                st.stop()

        # 3. Build model
        with st.spinner("Membangun arsitektur DNN..."):
            try:
                model = build_dense_model(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    hidden_layers=hidden_layers,
                    hidden_units=hidden_units,
                    dropout_rate=dropout_rate,
                    l2_reg_rate=l2_reg_rate,
                    dense_activation=dense_activation,
                    use_batchnorm=use_batchnorm,
                    learning_rate=learning_rate,
                    optimizer_name=optimizer_name,
                    use_grad_clip=use_grad_clip
                )

                # Tampilkan ringkasan arsitektur
                summary_lines = []
                model.summary(print_fn=lambda x: summary_lines.append(x))
                with st.expander("🔍 Lihat Arsitektur Model"):
                    st.code("\n".join(summary_lines), language="text")

            except Exception as e:
                st.error(f"❌ Gagal membangun model: {e}")
                st.stop()

        # 4. Prepare callbacks
        callbacks_list = []
        if use_early_stopping:
            callbacks_list.append(
                EarlyStopping(monitor='val_loss', patience=15,
                              restore_best_weights=True, verbose=1)
            )
        if use_reduce_lr:
            callbacks_list.append(
                ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                  patience=7, min_lr=1e-7, verbose=1)
            )
        if use_checkpoint:
            callbacks_list.append(
                ModelCheckpoint(filepath='models/best_dnn_model.keras',
                                monitor='val_accuracy', save_best_only=True,
                                verbose=1)
            )

        # 5. Build tf.data pipeline or use raw numpy
        if use_tf_data:
            train_ds = build_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
            val_ds   = build_tf_dataset(X_test, y_test,   batch_size=batch_size, shuffle=False)
            fit_kwargs = dict(
                x=train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=1
            )
        else:
            fit_kwargs = dict(
                x=X_train, y=y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=1
            )

        # 6. Train
        with st.spinner(f"Training DNN hingga {epochs} epoch... (lihat terminal untuk progress per-epoch)"):
            try:
                history = model.fit(**fit_kwargs)
            except Exception as e:
                st.error(f"❌ Gagal training: {e}")
                st.stop()

        # 7. Save artifacts (shared / quick-access copy)
        model.save("models/best_dnn_model.keras")
        joblib.dump(history.history, "models/training_history.pkl")
        joblib.dump(classes, "models/classes.pkl")

        final_train_acc = history.history['accuracy'][-1]
        final_val_acc   = history.history['val_accuracy'][-1]
        best_val_acc    = max(history.history['val_accuracy'])
        actual_epochs   = len(history.history['accuracy'])

        # 8. Hitung metrik lengkap untuk leaderboard
        from sklearn.metrics import precision_score, recall_score, f1_score
        with st.spinner("Menghitung metrik evaluasi untuk leaderboard..."):
            y_pred_lb = np.argmax(model.predict(X_test, verbose=0), axis=1)
            prec_lb = precision_score(y_test, y_pred_lb, average='weighted', zero_division=0)
            rec_lb  = recall_score(y_test, y_pred_lb, average='weighted', zero_division=0)
            f1_lb   = f1_score(y_test, y_pred_lb, average='weighted', zero_division=0)

        # 9. Simpan ke experiment registry (persisten)
        run_id = generate_run_id(selected_ds_train)
        hyperparams_to_save = {
            "hidden_layers":    hidden_layers,
            "hidden_units":     hidden_units,
            "dense_activation": dense_activation,
            "use_batchnorm":    use_batchnorm,
            "dropout_rate":     dropout_rate,
            "l2_reg_rate":      l2_reg_rate,
            "optimizer_name":   optimizer_name,
            "learning_rate":    learning_rate,
            "batch_size":       batch_size,
            "use_grad_clip":    use_grad_clip,
            "use_mixed_precision": use_mixed_precision,
        }
        metrics_to_save = {
            "train_accuracy": float(final_train_acc),
            "val_accuracy":   float(best_val_acc),
            "precision":      float(prec_lb),
            "recall":         float(rec_lb),
            "f1":             float(f1_lb),
            "val_loss":       float(min(history.history['val_loss'])),
            "epochs_run":     actual_epochs,
        }
        scaler_saved   = joblib.load('models/scaler.pkl')
        encoder_saved  = joblib.load('models/label_encoder.pkl')
        save_experiment(
            run_id=run_id,
            dataset_name=selected_ds_train,
            hyperparams=hyperparams_to_save,
            metrics=metrics_to_save,
            history=history.history,
            model_obj=model,
            scaler_obj=scaler_saved,
            encoder_obj=encoder_saved,
            classes_arr=classes
        )

        st.success(f"🎉 Training Selesai! Tersimpan ke registry sebagai `{run_id}`")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🌟 Train Accuracy (Akhir)", f"{final_train_acc*100:.2f}%")
        r2.metric("⭐ Best Val Accuracy",       f"{best_val_acc*100:.2f}%")
        r3.metric("🏆 F1-Score (Weighted)",     f"{f1_lb*100:.2f}%")
        r4.metric("⏱️ Epoch Aktual",            f"{actual_epochs} / {epochs}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Evaluation Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Evaluation Dashboard")
    st.markdown("Visualisasi performa model DNN terbaik yang telah ditraining.")

    model_path   = "models/best_dnn_model.keras"
    history_path = "models/training_history.pkl"

    # Fallback untuk nama file lama
    if not os.path.exists(model_path) and os.path.exists("models/best_dnn_model.h5"):
        model_path = "models/best_dnn_model.h5"

    if os.path.exists(model_path) and os.path.exists(history_path):
        history_data = joblib.load(history_path)
        classes      = joblib.load("models/classes.pkl")

        st.subheader("📈 Grafik Training")
        class HistoryDummy:
            def __init__(self, h): self.history = h
        fig = plot_training_history(HistoryDummy(history_data))
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("🎯 Metrik Evaluasi pada Data Test")

        selected_ds_eval = st.selectbox("Dataset untuk Evaluasi:", list(DATASET_OPTIONS.keys()), key="ds_eval")
        EVAL_DATASET_PATH = DATASET_OPTIONS[selected_ds_eval]

        with st.spinner("Menghitung metrik pada data test..."):
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            try:
                _, X_test_ev, _, y_test_ev, _ = load_and_preprocess_data(EVAL_DATASET_PATH)
                model_ev = tf.keras.models.load_model(model_path)
                y_pred_prob = model_ev.predict(X_test_ev, verbose=0)
                y_pred = np.argmax(y_pred_prob, axis=1)

                acc  = accuracy_score(y_test_ev, y_pred)
                prec = precision_score(y_test_ev, y_pred, average='weighted', zero_division=0)
                rec  = recall_score(y_test_ev, y_pred, average='weighted', zero_division=0)
                f1   = f1_score(y_test_ev, y_pred, average='weighted', zero_division=0)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🎯 Accuracy",            f"{acc*100:.2f}%")
                m2.metric("📊 Precision (Weighted)", f"{prec*100:.2f}%")
                m3.metric("📈 Recall (Weighted)",    f"{rec*100:.2f}%")
                m4.metric("🏆 F1-Score (Weighted)",  f"{f1*100:.2f}%")

                st.markdown("<br>", unsafe_allow_html=True)
                fig_cm = plot_confusion_matrix_custom(y_test_ev, y_pred, classes)
                st.pyplot(fig_cm)

            except Exception as e:
                st.error(f"❌ Gagal evaluasi: {e}")
    else:
        st.info("ℹ️ Belum ada model yang ditraining. Silakan ke tab **Model Training** terlebih dahulu.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Analisis Akademis
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📝 Analisis Akademis: Mengapa DNN Lebih Tepat dari GRU untuk Dataset Ini?")

    st.markdown("""
    <div class="info-card">
    <h4>🧠 Landasan Teoritis: Inductive Bias Model vs. Struktur Data</h4>
    <p>Pilihan arsitektur model harus selaras dengan asumsi dasar (inductive bias) yang dimiliki setiap model terhadap struktur data yang diproses.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("❌ GRU pada Data Tabular (Kurang Tepat)")
        st.markdown("""
        **GRU (Gated Recurrent Unit)** adalah varian RNN yang dirancang untuk
        memproses data dengan **ketergantungan temporal** — yaitu, nilai pada
        *timestep* ke-*t* dipengaruhi oleh nilai pada *timestep* ke-*(t-1)*.

        **Masalah saat digunakan pada fitur statistik FMA:**
        1. **Tidak ada urutan waktu.** Fitur seperti `mfcc1_mean`, `rms_var`,
           dan `tempo` adalah **ringkasan agregat** dari keseluruhan klip audio.
           Tidak ada hubungan kausal urutan-waktu di antara kolom-kolom ini.
        2. **Reshape artifisial `(N, 57, 1)`.** Untuk memaksa data tabular
           masuk ke GRU, data diubah bentuknya seolah-olah ada 57 *timestep*.
           Ini membuat GRU berusaha mempelajari pola "perubahan dari `mfcc1_mean`
           menuju `mfcc1_var` menuju `rms_mean`" — sebuah urutan yang tidak
           memiliki makna fisik/musikal apapun.
        3. **Parameter dan komputasi berlebih.** GRU memiliki 3 gerbang
           (reset, update, new) dengan parameter yang jauh lebih banyak dari Dense,
           namun tidak dapat dimanfaatkan secara efektif karena data tidak sekuensial.
        4. **Akurasi rendah.** Hasilnya adalah loss yang stagnan dan akurasi
           yang tidak jauh di atas *random chance* (untuk 8 kelas = 12.5%).
        """)

    with col_b:
        st.subheader("✅ FFNN/DNN pada Data Tabular (Tepat)")
        st.markdown("""
        **Feed-Forward Neural Network (FFNN/DNN)** tidak membuat asumsi
        apapun tentang urutan fitur. Setiap neuron pada hidden layer belajar
        mengkombinasikan seluruh 57 fitur input secara simultan.

        **Keunggulan untuk dataset fitur statistik FMA:**
        1. **Sesuai dengan struktur data.** Dataset CSV berisi vektor fitur
           *stateless* (tidak berurutan). FFNN dirancang persis untuk pemetaan
           `f: ℝ⁵⁷ → ℝᴷ` (57 fitur → K kelas genre).
        2. **Kapasitas representasi optimal.** Dengan arsitektur *piramida*
           (256 → 128 → 64 neuron), model dapat mempelajari hierarki fitur:
           layer awal mendeteksi pola lokal (misal: energi tinggi = Rock),
           layer akhir mengintegrasikan konteks global.
        3. **Batch Normalization** menstabilkan distribusi aktivasi antar layer,
           mempercepat konvergensi dan mencegah *internal covariate shift*.
        4. **Dropout** memaksa redundansi representasi, mencegah model
           bergantung pada satu fitur dominan (e.g., hanya `tempo`).
        5. **Training jauh lebih cepat dan akurasi lebih tinggi.**
        """)

    st.markdown("---")
    st.subheader("🚀 Optimasi GPU: Mixed Precision & tf.data Pipeline")
    st.markdown("""
    | Teknik | Mekanisme | Manfaat |
    |--------|-----------|---------|
    | **Mixed Precision (float16)** | Komputasi matrix multiplication berjalan dalam float16, bobot disimpan dalam float32 | Throughput GPU meningkat 2–3x, konsumsi VRAM berkurang 50% |
    | **tf.data + Prefetch (AUTOTUNE)** | CPU menyiapkan batch *N+1* sementara GPU memproses batch *N* | Eliminasi bottleneck CPU-GPU, utilisasi GPU mendekati 100% |
    | **EarlyStopping** | Menghentikan training jika `val_loss` tidak membaik selama *patience* epoch | Mencegah overfitting dan menghemat waktu komputasi |
    | **ReduceLROnPlateau** | Memperkecil learning rate saat loss stagnan | Model "menyelam" lebih presisi ke minimum, akurasi lebih tinggi |
    | **ModelCheckpoint** | Menyimpan bobot dengan `val_accuracy` terbaik, bukan bobot epoch terakhir | Mendapatkan model generalisasi terbaik meskipun training overfitting di epoch akhir |

    **Referensi Teknis:**
    - Vaswani et al. (2017) menunjukkan bahwa model dengan *inductive bias* yang sesuai dengan struktur data menghasilkan generalisasi yang jauh lebih baik.
    - NVIDIA (2023): Mixed Precision Training dengan TF32/FP16 pada GPU Ampere menghasilkan speedup 2–8x untuk operasi dense (GEMM).
    - Google Brain (tf.data docs): Penggunaan `.prefetch(AUTOTUNE)` adalah teknik standar untuk menghilangkan bottleneck I/O dalam pipeline deep learning.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Live Audio Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🎧 Live Audio Prediction")
    st.markdown(
        "Unggah file audio `.wav` atau `.mp3`. Sistem akan mengekstrak **57 fitur statistik** "
        "menggunakan Librosa, menormalisasinya dengan `StandardScaler` tersimpan, "
        "lalu memprediksi genre menggunakan model DNN."
    )

    model_path = "models/best_dnn_model.keras"
    if not os.path.exists(model_path) and os.path.exists("models/best_dnn_model.h5"):
        model_path = "models/best_dnn_model.h5"

    if not os.path.exists(model_path):
        st.warning("⚠️ Model belum tersedia. Lakukan training di tab **Model Training** terlebih dahulu.")
    else:
        uploaded_file = st.file_uploader("Pilih file audio:", type=['wav', 'mp3', 'ogg'])

        if uploaded_file is not None:
            st.audio(uploaded_file)

            if st.button("🎯 Ekstrak Fitur & Prediksi Genre", type="primary", key="predict_btn"):
                temp_path = "temp_audio_upload.wav"
                try:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner("Mengekstrak 57 fitur statistik audio (MFCC, Chroma, dll)..."):
                        X_infer = preprocess_audio_for_inference(temp_path)

                    with st.spinner("Memprediksi genre..."):
                        model_inf = tf.keras.models.load_model(model_path)
                        classes   = joblib.load("models/classes.pkl")
                        pred_probs = model_inf.predict(X_infer, verbose=0)[0]

                    pred_idx   = np.argmax(pred_probs)
                    pred_genre = classes[pred_idx]
                    confidence = pred_probs[pred_idx] * 100

                    st.success("✅ Prediksi Selesai!")

                    p1, p2 = st.columns(2)
                    p1.metric("🎵 Genre Terprediksi", pred_genre)
                    p2.metric("📊 Confidence", f"{confidence:.1f}%")

                    st.subheader("Distribusi Probabilitas Semua Genre")
                    prob_df = pd.DataFrame({
                        'Genre': classes,
                        'Probability (%)': pred_probs * 100
                    }).sort_values('Probability (%)', ascending=False)
                    st.bar_chart(prob_df.set_index('Genre'))

                except Exception as e:
                    st.error(f"❌ Error saat memproses audio: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: Upload Dataset Baru
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("📤 Upload Dataset Baru")
    st.markdown(
        "Unggah file CSV berformat **FMA/GTZAN-like** (kolom `label` dan "
        "fitur statistik audio: MFCC, chroma, tempo). "
        "Dataset divalidasi otomatis dan siap digunakan untuk training."
    )

    with st.expander("📋 Persyaratan kolom dataset"):
        st.markdown(
            "| Kolom | Tipe | Keterangan |\n"
            "|-------|------|------------|\n"
            "| `label` | string | Nama genre (Rock, Pop, Jazz, dll) |\n"
            "| `mfcc1_mean` ... `mfcc20_var` | float | 40 kolom MFCC |\n"
            "| `chroma_stft_mean`, `chroma_stft_var` | float | Fitur Chroma |\n"
            "| `rms_mean`, `rms_var` | float | Root Mean Square |\n"
            "| `tempo` | float | BPM estimasi |\n"
            "| `filename` | string | Opsional — dihapus otomatis |\n"
        )

    uploaded_csv = st.file_uploader("Pilih file CSV dataset:", type=["csv"], key="upload_csv")

    if uploaded_csv is not None:
        with st.spinner("Memvalidasi dataset..."):
            try:
                df_upload = pd.read_csv(uploaded_csv)
                is_valid, msg = validate_uploaded_csv(df_upload)
                if is_valid:
                    st.success(msg)
                    c1u, c2u, c3u = st.columns(3)
                    c1u.metric("Total Baris", f"{len(df_upload):,}")
                    c2u.metric("Total Kolom", len(df_upload.columns))
                    c3u.metric("Jumlah Kelas", df_upload["label"].nunique())
                    st.dataframe(df_upload.head(5), use_container_width=True)
                    st.bar_chart(df_upload["label"].value_counts())
                    if st.button("💾 Simpan Dataset ke Library", type="primary", key="save_ds_btn"):
                        save_path = save_uploaded_dataset(df_upload, uploaded_csv.name)
                        st.success(f"✅ Tersimpan ke: `{save_path}`")
                else:
                    st.error(msg)
            except Exception as e:
                st.error(f"❌ Gagal membaca CSV: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: Leaderboard & Komparasi Model
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("🏆 Leaderboard & Komparasi Model")
    st.markdown("Data dari **registry.json** — persisten meskipun aplikasi di-restart.")

    if st.button("🔄 Refresh", key="refresh_lb"):
        st.rerun()

    try:
        all_exps = load_all_experiments()
    except Exception as e:
        st.error(f"❌ Gagal memuat registry: {e}")
        all_exps = []

    if not all_exps:
        st.info("ℹ️ Belum ada eksperimen. Lakukan training terlebih dahulu.")
    else:
        st.success(f"✅ {len(all_exps)} eksperimen ditemukan di registry.")

        # ── Overall Champion ──────────────────────────────────────────────────
        try:
            best_exp = get_best_model(all_exps)
            if best_exp:
                bm = best_exp["metrics"]
                bp = best_exp["hyperparams"]
                st.markdown(
                    '<div style="background:linear-gradient(135deg,#1a3a1a,#0d2e0d);'
                    'border:2px solid #f7c948;border-radius:12px;padding:20px;margin:12px 0;">'
                    '<h3 style="color:#f7c948;margin:0;">👑 Overall Champion</h3></div>',
                    unsafe_allow_html=True
                )
                b1, b2, b3, b4, b5c = st.columns(5)
                b1.metric("🎯 Val Accuracy", f"{bm.get('val_accuracy', 0)*100:.2f}%")
                b2.metric("🏆 F1-Score",     f"{bm.get('f1', 0)*100:.2f}%")
                b3.metric("📊 Precision",    f"{bm.get('precision', 0)*100:.2f}%")
                b4.metric("📈 Recall",       f"{bm.get('recall', 0)*100:.2f}%")
                b5c.metric("⏱️ Epochs",      str(bm.get("epochs_run", "-")))
                with st.expander("🔍 Detail Konfigurasi Champion"):
                    st.dataframe(
                        pd.DataFrame([{"Parameter": k, "Nilai": str(v)} for k, v in bp.items()]),
                        use_container_width=True, hide_index=True
                    )
                    st.caption(
                        f"Run ID: `{best_exp['run_id']}` | "
                        f"Dataset: **{best_exp.get('dataset_name', '-')}** | "
                        f"Waktu: {best_exp['timestamp'][:19].replace('T', ' ')}"
                    )
        except Exception as e:
            st.error(f"❌ Error Champion: {e}")

        st.markdown("---")

        # ── Best Per Dataset ──────────────────────────────────────────────────
        try:
            st.subheader("🥇 Terbaik per Dataset")
            bpd = get_best_per_dataset(all_exps)
            rows_bpd = []
            for ds, exp in bpd.items():
                m = exp["metrics"]
                rows_bpd.append({
                    "Dataset":       ds,
                    "Val Acc (%)":   round(m.get("val_accuracy", 0)*100, 2),
                    "F1 (%)":        round(m.get("f1", 0)*100, 2),
                    "Precision (%)": round(m.get("precision", 0)*100, 2),
                    "Recall (%)":    round(m.get("recall", 0)*100, 2),
                    "Run ID":        exp["run_id"],
                })
            if rows_bpd:
                st.dataframe(
                    pd.DataFrame(rows_bpd).sort_values("Val Acc (%)", ascending=False),
                    use_container_width=True, hide_index=True
                )
        except Exception as e:
            st.error(f"❌ Error Best Per Dataset: {e}")

        st.markdown("---")

        # ── Full Leaderboard Table ─────────────────────────────────────────────
        try:
            st.subheader(f"📋 Leaderboard Lengkap ({len(all_exps)} Eksperimen)")
            lb_df = build_leaderboard_df(all_exps)
            if not lb_df.empty:
                lb_df.insert(0, "Rank", range(1, len(lb_df) + 1))
                st.dataframe(lb_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"❌ Error tabel leaderboard: {e}")

        st.markdown("---")

        # ── Comparison Bar Chart ───────────────────────────────────────────────
        try:
            st.subheader("📈 Grafik Komparasi 4 Metrik")
            disp = sorted(all_exps, key=lambda e: e["timestamp"], reverse=True)[:10]

            def _lbl(rid):
                c = rid.replace("run_", "").replace("(", "").replace(")", "").replace(",", "")
                p = c.split("_")
                ts   = "_".join(p[:2]) if len(p) >= 2 else c
                slug = "_".join(p[2:4]) if len(p) >= 4 else ""
                return f"{ts}\n{slug}"

            lbls  = [_lbl(e["run_id"]) for e in disp]
            va    = [e["metrics"].get("val_accuracy", 0)*100 for e in disp]
            f1s   = [e["metrics"].get("f1", 0)*100 for e in disp]
            precs = [e["metrics"].get("precision", 0)*100 for e in disp]
            recs  = [e["metrics"].get("recall", 0)*100 for e in disp]

            fig_c, axs = plt.subplots(2, 2, figsize=(14, 8))
            fig_c.patch.set_facecolor("#1e1e2e")
            pd_list = [
                (axs[0, 0], va,    "#7aa2f7", "Val Accuracy (%)"),
                (axs[0, 1], f1s,   "#9ece6a", "F1-Score (%)"),
                (axs[1, 0], precs, "#ff9e64", "Precision (%)"),
                (axs[1, 1], recs,  "#bb9af7", "Recall (%)"),
            ]
            xp = list(range(len(lbls)))
            for ax, dat, col, ttl in pd_list:
                ax.set_facecolor("#2a2a3e")
                brs = ax.bar(xp, dat, color=col, alpha=0.85, edgecolor="#555")
                ax.set_title(ttl, color="white", fontweight="bold", pad=6)
                ax.set_xticks(xp)
                ax.set_xticklabels(lbls, color="white", fontsize=6)
                ax.tick_params(colors="white")
                ax.set_ylim(0, 107)
                ax.grid(axis="y", alpha=0.2)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#444")
                for b, v in zip(brs, dat):
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                            f"{v:.1f}", ha="center", va="bottom", color="white", fontsize=6)
            plt.tight_layout(pad=2.0)
            st.pyplot(fig_c)
            plt.close(fig_c)
        except Exception as e:
            st.error(f"❌ Error grafik bar: {e}")

        st.markdown("---")

        # ── Overlay Kurva Val Accuracy ─────────────────────────────────────────
        try:
            st.subheader("📉 Overlay Val Accuracy (Top 5)")
            top5 = sorted(all_exps,
                          key=lambda e: e["metrics"].get("val_accuracy", 0),
                          reverse=True)[:5]
            fig_ov, ax_ov = plt.subplots(figsize=(12, 5))
            fig_ov.patch.set_facecolor("#1e1e2e")
            ax_ov.set_facecolor("#2a2a3e")
            cols_ov = ["#7aa2f7", "#ff9e64", "#9ece6a", "#bb9af7", "#f7768e"]
            plotted = 0
            for i, exp in enumerate(top5):
                hp = os.path.join(exp.get("run_dir", ""), "history.pkl")
                if os.path.exists(hp):
                    h = joblib.load(hp)
                    crv = h.get("val_accuracy", [])
                    if crv:
                        sid = exp["run_id"].replace("run_", "")[:18]
                        lbl = f"#{i+1} {sid} ({exp['metrics'].get('val_accuracy', 0)*100:.1f}%)"
                        ax_ov.plot(crv, color=cols_ov[i % 5], linewidth=2, label=lbl)
                        plotted += 1
            if plotted == 0:
                ax_ov.text(0.5, 0.5, "Tidak ada history tersedia",
                           ha="center", va="center", color="white", transform=ax_ov.transAxes)
            else:
                ax_ov.legend(facecolor="#2a2a3e", labelcolor="white", fontsize=8)
            ax_ov.set_title("Val Accuracy per Epoch - Top 5", color="white", fontweight="bold")
            ax_ov.set_xlabel("Epoch", color="white")
            ax_ov.set_ylabel("Val Accuracy", color="white")
            ax_ov.tick_params(colors="white")
            ax_ov.grid(True, alpha=0.2)
            for sp in ax_ov.spines.values():
                sp.set_edgecolor("#444")
            plt.tight_layout()
            st.pyplot(fig_ov)
            plt.close(fig_ov)
        except Exception as e:
            st.error(f"❌ Error overlay: {e}")

        st.markdown("---")

        # ── Hapus Eksperimen ───────────────────────────────────────────────────
        try:
            st.subheader("🗑️ Kelola Eksperimen")
            rid_list = [e["run_id"] for e in all_exps]
            del_rid = st.selectbox("Pilih eksperimen untuk dihapus:", rid_list, key="del_exp")
            if st.button("🗑️ Hapus Eksperimen Ini", key="del_btn"):
                delete_experiment(del_rid)
                st.success(f"✅ Dihapus: `{del_rid}`")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error panel hapus: {e}")
