import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
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
    st.header("Model Training & Experimentation")
    st.markdown("Eksperimen dengan berbagai hyperparameter untuk mencari kinerja terbaik.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hyperparameters")
        gru_layers = st.number_input("Jumlah Layer GRU", min_value=1, max_value=5, value=2)
        gru_units = st.selectbox("Jumlah Unit GRU per Layer", [16, 32, 64, 128, 256], index=2)
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.9, 0.2, 0.1)
        learning_rate = st.selectbox("Learning Rate", [0.01, 0.001, 0.0001, 0.00001], index=1)
        
    with col2:
        st.subheader("Training Settings")
        optimizer_name = st.selectbox("Optimizer", ['adam', 'rmsprop', 'sgd'])
        activation_fn = st.selectbox("Activation Function (GRU)", ['relu', 'tanh', 'tanh'])
        loss_fn = st.selectbox("Loss Function", ['sparse_categorical_crossentropy'])
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        epochs = st.number_input("Jumlah Epoch", min_value=1, max_value=200, value=20)
        
    if st.button("Mulai Training Model"):
        with st.spinner("Memuat dan memproses dataset..."):
            X_train, X_test, y_train, y_test, classes = load_and_preprocess_data(DATASET_PATH)
            num_classes = len(classes)
            input_shape = (X_train.shape[1], X_train.shape[2])
            
        with st.spinner("Membangun model GRU..."):
            model = build_gru_model(
                input_shape=input_shape,
                num_classes=num_classes,
                gru_layers=gru_layers,
                gru_units=gru_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                optimizer_name=optimizer_name,
                activation=activation_fn,
                loss_function=loss_fn
            )
            
        with st.spinner(f"Training model selama {epochs} epoch..."):
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Save the best model and history
            model.save("models/best_gru_model.h5")
            joblib.dump(history.history, "models/training_history.pkl")
            joblib.dump(classes, "models/classes.pkl")
            
            st.success("Training Selesai! Model disimpan sebagai `best_gru_model.h5`")
            
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            st.metric("Final Training Accuracy", f"{final_train_acc:.4f}")
            st.metric("Final Validation Accuracy", f"{final_val_acc:.4f}")

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
    ### Faktor yang Mempengaruhi Performa Model GRU
    
    Berdasarkan arsitektur yang disediakan, berikut adalah penjelasan masing-masing parameter terhadap kinerja model:
    
    1. **Activation Function**: Mengubah aktivasi GRU (standarnya adalah `tanh`). Penggunaan `relu` dapat mengatasi vanishing gradient pada data sequence yang panjang, namun terkadang kurang stabil dibandingkan `tanh` pada RNN.
    2. **Optimizer**: `Adam` biasanya memberikan konvergensi yang lebih cepat dan performa terbaik secara umum berkat mekanisme momentum dan adaptasi learning rate. `RMSprop` juga sangat direkomendasikan untuk arsitektur RNN/GRU.
    3. **Loss Function**: Karena kita melakukan klasifikasi multikelas (10 genre), `sparse_categorical_crossentropy` adalah fungsi loss yang tepat.
    4. **Learning Rate**: Learning rate yang terlalu besar (misal 0.01) membuat model melompati minima lokal, menyebabkan loss fluktuatif. Learning rate yang lebih kecil (0.001 atau 0.0001) menghasilkan konvergensi yang lebih stabil.
    5. **Batch Size**: Batch size kecil (16, 32) memberikan generalisasi yang lebih baik dan update bobot yang lebih sering, namun memakan waktu training yang lebih lama.
    6. **Jumlah Epoch**: Terlalu sedikit epoch menyebabkan underfitting. Terlalu banyak epoch menyebabkan overfitting, ditandai dengan *train accuracy* yang tinggi namun *val accuracy* yang menurun.
    7. **Jumlah Layer GRU**: Menambah layer GRU (stacked GRU) memungkinkan model menangkap pola hierarkis yang lebih kompleks dalam sekuens data. Namun, komputasi menjadi lebih berat dan berisiko overfitting jika dataset tidak cukup besar.
    8. **Dropout**: Regularisasi *Dropout* mematikan sebagian neuron secara acak selama training, sangat krusial untuk mencegah model sekadar menghafal data training (overfitting).
    
    ### Kesimpulan Kinerja Model
    Model GRU terbukti efektif untuk klasifikasi genre musik karena kemampuannya mengingat informasi "masa lalu" dari rentetan fitur berurutan (sequence features). Kombinasi **Stacked GRU (2 layer)** dengan **Dropout 0.2** dan optimizer **Adam** umumnya memberikan *sweet spot* antara akurasi yang baik dan generalisasi pada data pengujian.
    """)

# --- TAB 5: Live Audio Prediction ---
with tab5:
    st.header("Live Audio Prediction")
    st.markdown("Unggah file audio `.wav` atau `.mp3` untuk mengekstrak fitur secara real-time dan memprediksi genre.")
    
    uploaded_file = st.file_uploader("Pilih file audio", type=['wav', 'mp3', 'ogg'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Ekstrak Fitur & Prediksi Genre"):
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
