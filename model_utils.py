import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, SGD, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def enable_mixed_precision():
    """
    Mengaktifkan Mixed Precision (float16/float32) untuk mempercepat komputasi GPU.
    Boboti (weights) tetap disimpan dalam float32 untuk stabilitas numerik,
    tetapi operasi komputasi berjalan dalam float16 yang jauh lebih cepat di GPU modern.
    """
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"[Mixed Precision] Compute dtype: {policy.compute_dtype}")
    print(f"[Mixed Precision] Variable dtype: {policy.variable_dtype}")


def build_dense_model(input_dim, num_classes,
                      hidden_layers=3,
                      hidden_units=256,
                      dropout_rate=0.3,
                      l2_reg_rate=0.001,
                      dense_activation='relu',
                      use_batchnorm=True,
                      learning_rate=0.001,
                      optimizer_name='Adam',
                      use_grad_clip=False):
    """
    Membangun Feed-Forward Neural Network (FFNN) / Deep Neural Network (DNN)
    yang dioptimalkan untuk data tabular fitur statistik audio.

    Arsitektur ini lebih tepat dibandingkan GRU untuk dataset CSV karena:
    - Input: vektor fitur statis (mean, variance) -- bukan urutan waktu.
    - GRU mengharapkan pola sekuensial antar timestep; data tabular tidak memiliki
      hubungan kausal antar kolom semacam itu.
    - Dense layers cukup kuat untuk mempelajari batas keputusan non-linear
      dari ruang fitur statistik berdimensi tinggi.

    Args:
        input_dim (int): Jumlah fitur input (kolom pada CSV setelah pra-proses).
        num_classes (int): Jumlah kelas genre musik.
        hidden_layers (int): Jumlah hidden layer Dense.
        hidden_units (int): Jumlah neuron per hidden layer.
        dropout_rate (float): Probabilitas dropout (0 = dinonaktifkan).
        l2_reg_rate (float): Koefisien L2 regularization.
        dense_activation (str): Fungsi aktivasi untuk hidden layer.
        use_batchnorm (bool): Mengaktifkan Batch Normalization antar layer.
        learning_rate (float): Learning rate optimizer.
        optimizer_name (str): Nama optimizer yang digunakan.
        use_grad_clip (bool): Mengaktifkan gradient clipping (clipvalue=1.0).

    Returns:
        model: tf.keras.Model yang sudah dikompilasi.
    """
    kernel_reg = tf.keras.regularizers.l2(l2_reg_rate) if l2_reg_rate > 0.0 else None

    model = Sequential(name="FFNN_Music_Genre_Classifier")
    model.add(Input(shape=(input_dim,)))

    for i in range(hidden_layers):
        # Jumlah unit dikurangi secara bertahap (Pyramid Architecture)
        units = max(hidden_units // (2 ** i), 64)
        model.add(Dense(units, activation=dense_activation,
                        kernel_regularizer=kernel_reg,
                        name=f"hidden_{i+1}"))

        if use_batchnorm:
            model.add(BatchNormalization(name=f"batchnorm_{i+1}"))

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate, name=f"dropout_{i+1}"))

    # Output layer — softmax + float32 cast agar kompatibel dengan mixed_float16
    model.add(Dense(num_classes, activation='linear', name="output_logits"))
    model.add(tf.keras.layers.Activation('softmax', dtype='float32', name="output_softmax"))

    # --- Optimizer Selection ---
    opt_kwargs = {'learning_rate': learning_rate}
    if use_grad_clip:
        opt_kwargs['clipvalue'] = 1.0

    opt_name = optimizer_name.lower()
    optimizers_map = {
        'adam': Adam,
        'adamw': AdamW,
        'rmsprop': RMSprop,
        'sgd': SGD,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adamax': Adamax,
        'nadam': Nadam,
    }
    optimizer_cls = optimizers_map.get(opt_name, Adam)
    optimizer = optimizer_cls(**opt_kwargs)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_tf_dataset(X, y, batch_size=64, shuffle=False):
    """
    Membungkus numpy array ke dalam tf.data.Dataset yang dioptimalkan untuk GPU.

    Penggunaan .prefetch(tf.data.AUTOTUNE) memastikan GPU tidak menganggur
    menunggu CPU selesai menyiapkan batch berikutnya (overlapping CPU-GPU pipeline).

    Args:
        X (np.ndarray): Array fitur input.
        y (np.ndarray): Array label target.
        batch_size (int): Ukuran batch per step.
        shuffle (bool): Mengacak data (gunakan True untuk data training).

    Returns:
        tf.data.Dataset: Dataset yang sudah dibatch dan diprefetch.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def plot_training_history(history):
    """Memplot grafik akurasi dan loss dari riwayat training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1e1e2e')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#2a2a3e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#7aa2f7', linewidth=2)
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff9e64', linewidth=2, linestyle='--')
    ax1.set_title('Model Accuracy', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(facecolor='#2a2a3e', labelcolor='white')
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train Loss', color='#f7768e', linewidth=2)
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Val Loss', color='#9ece6a', linewidth=2, linestyle='--')
    ax2.set_title('Model Loss', fontweight='bold')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(facecolor='#2a2a3e', labelcolor='white')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix_custom(y_true, y_pred, classes):
    """Memplot confusion matrix dengan style yang lebih informatif."""
    cm = confusion_matrix(y_true, y_pred)
    # Normalisasi untuk menampilkan persentase
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor='#333')
    ax.set_title('Confusion Matrix (Normalized)', color='white', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', color='white')
    ax.set_xlabel('Predicted Label', color='white')
    ax.tick_params(colors='white')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig
