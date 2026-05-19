import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, SGD, Adagrad, Adadelta, Adamax, Nadam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def get_activation_layer(activation_name):
    """Helper function to return Keras activation layer or string."""
    if activation_name.lower() == 'leaky_relu':
        return tf.keras.layers.LeakyReLU()
    return activation_name.lower()

def build_gru_model(input_shape, num_classes, 
                    gru_layers=1, gru_units=64, 
                    dropout_rate=0.2, recurrent_dropout_rate=0.0,
                    l2_reg_rate=0.0,
                    gru_activation='tanh', dense_activation='relu',
                    use_bidirectional=False, use_batchnorm=False,
                    flattening_type='Gunakan Hidden State Terakhir Saja',
                    learning_rate=0.001, optimizer_name='Adam', use_grad_clip=False):
    
    model = Sequential()
    
    # Setup Regularizer
    kernel_reg = tf.keras.regularizers.l2(l2_reg_rate) if l2_reg_rate > 0.0 else None
    
    # Process Activation
    gru_act = get_activation_layer(gru_activation)
    dense_act = get_activation_layer(dense_activation)
    
    for i in range(gru_layers):
        is_last_layer = (i == gru_layers - 1)
        if not is_last_layer:
            return_seq = True
        else:
            if flattening_type in ['Global Average Pooling 1D', 'Global Max Pooling 1D', 'Flatten']:
                return_seq = True
            else:
                return_seq = False
                
        # Base GRU layer
        gru_layer = GRU(
            units=gru_units, 
            return_sequences=return_seq, 
            activation=gru_act,
            recurrent_dropout=recurrent_dropout_rate,
            kernel_regularizer=kernel_reg
        )
        
        # Add Input shape to the first layer
        if i == 0:
            if use_bidirectional:
                model.add(Bidirectional(gru_layer, input_shape=input_shape))
            else:
                # Need to recreate GRU to directly pass input_shape for non-bidirectional
                gru_layer = GRU(
                    units=gru_units, 
                    return_sequences=return_seq, 
                    activation=gru_act,
                    recurrent_dropout=recurrent_dropout_rate,
                    kernel_regularizer=kernel_reg,
                    input_shape=input_shape
                )
                model.add(gru_layer)
        else:
            if use_bidirectional:
                model.add(Bidirectional(gru_layer))
            else:
                model.add(gru_layer)
                
        # Batch Normalization
        if use_batchnorm:
            model.add(BatchNormalization())
            
        # Dropout
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
            
    # Flattening Layer
    if flattening_type == 'Global Average Pooling 1D':
        model.add(GlobalAveragePooling1D())
    elif flattening_type == 'Global Max Pooling 1D':
        model.add(GlobalMaxPooling1D())
    elif flattening_type == 'Flatten':
        model.add(Flatten())
        
    # Intermediate Dense Layer for the requested Dense Activation
    model.add(Dense(64, activation=dense_act, kernel_regularizer=kernel_reg))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
        
    # Output Layer (Softmax for multi-class)
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=kernel_reg))
    
    # Optimizer selection and Gradient Clipping
    opt_kwargs = {'learning_rate': learning_rate}
    if use_grad_clip:
        opt_kwargs['clipvalue'] = 1.0
        
    opt_name = optimizer_name.lower()
    if opt_name == 'adam':
        optimizer = Adam(**opt_kwargs)
    elif opt_name == 'adamw':
        optimizer = AdamW(**opt_kwargs)
    elif opt_name == 'rmsprop':
        optimizer = RMSprop(**opt_kwargs)
    elif opt_name == 'sgd':
        optimizer = SGD(**opt_kwargs)
    elif opt_name == 'adagrad':
        optimizer = Adagrad(**opt_kwargs)
    elif opt_name == 'adadelta':
        optimizer = Adadelta(**opt_kwargs)
    elif opt_name == 'adamax':
        optimizer = Adamax(**opt_kwargs)
    elif opt_name == 'nadam':
        optimizer = Nadam(**opt_kwargs)
    else:
        optimizer = Adam(**opt_kwargs)
        
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
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
