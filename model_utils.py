import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def build_gru_model(input_shape, num_classes, gru_layers=1, gru_units=64, 
                    dropout_rate=0.2, learning_rate=0.001, 
                    optimizer_name='adam', activation='relu', loss_function='sparse_categorical_crossentropy'):
    
    model = Sequential()
    
    # Input GRU Layer
    return_sequences = gru_layers > 1
    model.add(GRU(gru_units, input_shape=input_shape, return_sequences=return_sequences, activation=activation))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
        
    # Additional GRU Layers
    for i in range(1, gru_layers):
        return_sequences = i < gru_layers - 1
        model.add(GRU(gru_units, return_sequences=return_sequences, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
            
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Optimizer selection
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)
        
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy'])
    
    return model

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss', color='red')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Val Loss', color='green')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    return fig

def plot_confusion_matrix_custom(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
