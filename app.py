import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_utils import load_and_preprocess_data, preprocess_audio_for_inference
from model_utils import build_gru_model, plot_training_history, plot_confusion_matrix_custom
import joblib

st.set_page_config(page_title="Music Genre Classification", layout="wide")

st.title("🎵 Music Genre Classification with GRU")
st.markdown("Aplikasi Implementasi Deep Learning untuk Klasifikasi Genre Musik menggunakan Arsitektur Gated Recurrent Unit (GRU).")

# Define dataset path
DATASET_PATH = "FMA_like_GTZAN_for_Music_Genre_Classification/fma_3secs.csv"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Dataset Overview", 
    "⚙️ Model Training", 
    "📊 Evaluation Dashboard", 
    "📝 Discussion & Analysis", 
    "🎧 Live Audio Prediction"
])

# --- TAB 1: Dataset Overview ---
with tab1:
    st.header("Dataset Overview")
    st.write(f"Menggunakan dataset dari: `{DATASET_PATH}`")
    
    try:
        df_preview = pd.read_csv(DATASET_PATH, nrows=100)
        st.dataframe(df_preview.head())
        st.write(f"**Total Baris Preview:** {len(df_preview)}")
        
        st.subheader("Distribusi Label (Preview)")
        st.bar_chart(df_preview['label'].value_counts())
    except FileNotFoundError:
        st.error(f"Dataset tidak ditemukan di path: {DATASET_PATH}")

# --- TAB 2: Model Training ---
with tab2:
    st.header("Model Training & Experimentation (Advanced Hyperparameter Tuning)")
    st.markdown("Sesuaikan arsitektur GRU dan parameter training untuk mencapai performa terbaik sesuai kebutuhan.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Arsitektur Makro")
        gru_layers = st.number_input("Jumlah Layer GRU", min_value=1, max_value=5, value=2)
        gru_units = st.selectbox("Jumlah Unit GRU per Layer", [16, 32, 64, 128, 256], index=2)
        use_bidirectional = st.checkbox("Gunakan Bidirectional GRU", value=False)
        use_batchnorm = st.checkbox("Tambahkan Batch Normalization", value=False)
        flattening_type = st.selectbox("Lapisan Perataan (Output GRU ke Dense)", 
                                       ['Global Average Pooling 1D', 'Global Max Pooling 1D', 'Flatten', 'Gunakan Hidden State Terakhir Saja'])

        st.subheader("2. Fungsi Aktivasi")
        gru_activation = st.selectbox("Aktivasi Internal Layer GRU", ['tanh', 'relu', 'leaky_relu', 'elu', 'selu', 'gelu'])
        dense_activation = st.selectbox("Aktivasi Dense Layer", ['relu', 'leaky_relu', 'elu', 'selu', 'swish', 'linear'])

    with col2:
        st.subheader("3. Regularisasi & Pencegahan Overfitting")
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.7, 0.2, 0.1)
        recurrent_dropout_rate = st.slider("Recurrent Dropout (Internal GRU)", 0.0, 0.5, 0.0, 0.1)
        l2_reg_rate = st.selectbox("L2 Regularization (Kernel Weight Decay)", [0.0, 0.01, 0.001, 0.0001], format_func=lambda x: "0 (None)" if x == 0.0 else str(x))

        st.subheader("4. Optimasi & Stabilisator")
        optimizer_name = st.selectbox("Pilihan Optimizer", ['Adam', 'AdamW', 'RMSprop', 'SGD', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'])
        learning_rate = st.selectbox("Learning Rate", [0.01, 0.001, 0.0001, 0.00001], index=1)
        use_grad_clip = st.checkbox("Aktifkan Gradient Clipping (clipvalue=1.0)", value=False)

    with col3:
        st.subheader("5. Training Settings")
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        epochs = st.number_input("Jumlah Epoch", min_value=1, max_value=200, value=20)
        
        st.subheader("6. Otomatisasi Training (Callbacks)")
        use_early_stopping = st.checkbox("Aktifkan Early Stopping (patience=10)")
        use_reduce_lr = st.checkbox("Aktifkan ReduceLROnPlateau (patience=5, factor=0.2)")
        
    if st.button("Mulai Training Model", type="primary"):
        with st.spinner("Memuat dan memproses dataset..."):
            try:
                X_train, X_test, y_train, y_test, classes = load_and_preprocess_data(DATASET_PATH)
                num_classes = len(classes)
                input_shape = (X_train.shape[1], X_train.shape[2])
            except Exception as e:
                st.error(f"Gagal memuat dataset: {e}")
                st.stop()
            
        with st.spinner("Membangun model GRU..."):
            try:
                model = build_gru_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    gru_layers=gru_layers,
                    gru_units=gru_units,
                    dropout_rate=dropout_rate,
                    recurrent_dropout_rate=recurrent_dropout_rate,
                    l2_reg_rate=l2_reg_rate,
                    gru_activation=gru_activation,
                    dense_activation=dense_activation,
                    use_bidirectional=use_bidirectional,
                    use_batchnorm=use_batchnorm,
                    flattening_type=flattening_type,
                    learning_rate=learning_rate,
                    optimizer_name=optimizer_name,
                    use_grad_clip=use_grad_clip
                )
                
                # Prepare callbacks
                callbacks_list = []
                if use_early_stopping:
                    callbacks_list.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))
                if use_reduce_lr:
                    callbacks_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6))
                    
            except Exception as e:
                st.error(f"Gagal membangun model: {e}")
                st.stop()
            
        with st.spinner(f"Training model selama {epochs} epoch... (Lihat terminal untuk progress)"):
            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks_list,
                    verbose=1
                )
                
                # Save the best model and history
                model.save("models/best_gru_model.h5")
                joblib.dump(history.history, "models/training_history.pkl")
                joblib.dump(classes, "models/classes.pkl")
                
                st.success("Training Selesai! Model berhasil disimpan sebagai `best_gru_model.h5`")
                
                final_train_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                st.metric("Final Training Accuracy", f"{final_train_acc:.4f}")
                st.metric("Final Validation Accuracy", f"{final_val_acc:.4f}")
            except Exception as e:
                st.error(f"Gagal melakukan training: {e}")

# --- TAB 3: Evaluation Dashboard ---
with tab3:
    st.header("Evaluation Dashboard")
    st.markdown("Visualisasi performa model terbaik yang telah ditraining.")
    
    if os.path.exists("models/best_gru_model.h5") and os.path.exists("models/training_history.pkl"):
        history_data = joblib.load("models/training_history.pkl")
        classes = joblib.load("models/classes.pkl")
        
        st.subheader("Grafik Performa Model")
        # Dummy object to mimic Keras history object
        class HistoryDummy:
            def __init__(self, history):
                self.history = history
        
        fig = plot_training_history(HistoryDummy(history_data))
        st.pyplot(fig)
        
        st.subheader("Confusion Matrix")
        with st.spinner("Menghitung Confusion Matrix pada data testing..."):
            X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(DATASET_PATH)
            model = tf.keras.models.load_model("models/best_gru_model.h5")
            y_pred_prob = model.predict(X_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
            
            fig_cm = plot_confusion_matrix_custom(y_test, y_pred, classes)
            st.pyplot(fig_cm)
            
    else:
        st.info("Belum ada model yang ditraining. Silakan ke tab 'Model Training' terlebih dahulu.")

# --- TAB 4: Discussion & Analysis ---
with tab4:
    st.header("Hasil Diskusi & Analisis Eksperimen")
    
    st.markdown("""
    ### Analisis Dampak Hyperparameter (Hyperparameter Tuning)
    
    Aplikasi ini mendukung eksperimen tingkat lanjut untuk klasifikasi genre musik menggunakan model GRU. Berikut adalah panduan analisis masing-masing parameter:
    
    1. **Arsitektur Makro (Bidirectional & Batch Norm)**:
       - **Bidirectional GRU**: Memproses sekuens audio secara maju (forward) dan mundur (backward). Sangat kuat dalam memahami konteks genre dari struktur instrumen, namun memakan waktu komputasi 2x lipat.
       - **Batch Normalization**: Menstabilkan distribusi input antar layer, sehingga mempercepat konvergensi dan meminimalisir nilai loss yang *nan*. Disarankan aktif jika menggunakan layer GRU yang dalam (>2 layer).
       - **Lapisan Perataan (Flattening)**: Penggunaan *Global Average Pooling* seringkali memberikan generalisasi yang lebih baik dari *Flatten*, karena mengekstrak rata-rata setiap feature map secara komprehensif, mencegah overfitting yang drastis.
    
    2. **Fungsi Aktivasi**:
       - **Aktivasi GRU (tanh vs relu vs gelu)**: *tanh* adalah standar RNN karena outputnya terikat [-1, 1] yang stabil. Jika menggunakan *relu* atau turunannya (*leaky_relu*, *gelu*), sangat disarankan mengaktifkan **Gradient Clipping** untuk menghindari Exploding Gradient.
       
    3. **Regularisasi (L2 & Dropout)**:
       - **Dropout & Recurrent Dropout**: Dropout biasa mematikan neuron setelah output GRU, sementara Recurrent Dropout mematikannya *di dalam* step waktu (time-steps). Keduanya mencegah model "menghafal" (overfitting).
       - **L2 Regularization (Weight Decay)**: Memberikan pinalti terhadap bobot yang terlalu besar, memastikan tidak ada satu fitur (seperti tempo atau mfcc tertentu) yang terlalu mendominasi prediksi.
    
    4. **Optimasi & Stabilisator**:
       - **AdamW & Nadam**: Seringkali lebih superior dibandingkan Adam biasa dalam tugas audio, karena Nadam menambahkan *Nesterov Momentum* sementara AdamW memisahkan *weight decay*.
       - **Gradient Clipping**: Batas nilai gradien (clipvalue=1.0) menjadi penyelamat agar *loss function* tidak membesar tak terkendali.
       
    5. **Automasi Callbacks**:
       - **Early Stopping**: Sangat membantu untuk menghentikan epoch jika model sudah optimal dan loss validasi mulai naik (indikasi awal overfitting).
       - **ReduceLROnPlateau**: Memperkecil langkah belajar (learning rate) saat pergerakan loss mulai stagnan, membantu model "menyelam" lebih presisi menuju global minima.
    """)

# --- TAB 5: Live Audio Prediction ---
with tab5:
    st.header("Live Audio Prediction")
    st.markdown("Unggah file audio `.wav` atau `.mp3` untuk mengekstrak fitur secara real-time dan memprediksi genre.")
    
    uploaded_file = st.file_uploader("Pilih file audio", type=['wav', 'mp3', 'ogg'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Ekstrak Fitur & Prediksi Genre", type="primary"):
            if not os.path.exists("models/best_gru_model.h5"):
                st.error("Model belum ditraining! Harap lakukan training terlebih dahulu di Tab 'Model Training'.")
            else:
                with st.spinner("Mengekstrak fitur audio (MFCC, dll) menggunakan Librosa..."):
                    # Save temporarily
                    with open("temp_audio.wav", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Extract and preprocess
                        X_infer = preprocess_audio_for_inference("temp_audio.wav")
                        
                        # Predict
                        model = tf.keras.models.load_model("models/best_gru_model.h5")
                        classes = joblib.load("models/classes.pkl")
                        
                        pred_probs = model.predict(X_infer)[0]
                        pred_idx = np.argmax(pred_probs)
                        pred_genre = classes[pred_idx]
                        confidence = pred_probs[pred_idx] * 100
                        
                        st.success("Ekstraksi dan Prediksi Selesai!")
                        st.subheader(f"Genre Terprediksi: **{pred_genre}**")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                        
                        # Plot probabilities
                        prob_df = pd.DataFrame({
                            'Genre': classes,
                            'Probability (%)': pred_probs * 100
                        })
                        st.bar_chart(prob_df.set_index('Genre'))
                        
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses audio: {e}")
                    finally:
                        if os.path.exists("temp_audio.wav"):
                            os.remove("temp_audio.wav")
