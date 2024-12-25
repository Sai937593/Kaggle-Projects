import os
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime

from utils.tokenizer import TokenizerManager
from utils.data import DataManager
from models.classifier import Transformer_Classifier

def get_callbacks(model_name):
    """
    Create training callbacks
    """
    # Create logs and models directories
    log_dir = os.path.join('logs', model_name)
    model_dir = os.path.join('models', model_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        ),
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            update_freq='epoch'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        )
    ]

def plot_training_history(history, model_name):
    """
    Plot and save training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join('plots', model_name)
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'training_history.png'))
    plt.close()

def train_model(
    train_dataset,
    eval_dataset,
    vocab_size,
    max_length,
    strategy,
    model_name=None,
    batch_size=32,
    epochs=10,
    **model_kwargs
):
    """
    Train model using distribution strategy
    """
    if model_name is None:
        model_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create model and compile within strategy scope
    with strategy.scope():
        model = Transformer_Classifier(
            vocab_size=vocab_size,
            max_length=max_length,
            **model_kwargs
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.AUC(name='auc')
            ]
        )

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=epochs,
        callbacks=get_callbacks(model_name)
    )

    # Plot training history
    plot_training_history(history, model_name)
    
    return model, history

def main():
    # Set up distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    # Configuration
    CONFIG = {
        'vocab_size': 10000,
        'max_length': 128,
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'dropout_rate': 0.1,
        'batch_size': 32 * strategy.num_replicas_in_sync,  # Adjust batch size for multiple GPUs
        'epochs': 10
    }

    # Load data
    print("Loading data...")
    df = pd.read_csv('path/to/your/data.csv')  # Update path
    
    # Initialize tokenizer and data managers
    tokenizer_manager = TokenizerManager(
        vocab_size=CONFIG['vocab_size'],
        max_length=CONFIG['max_length']
    )
    
    data_manager = DataManager(
        tokenizer_manager,
        batch_size=CONFIG['batch_size']
    )

    # Prepare data
    tokenizer = tokenizer_manager.load_or_create_tokenizer(force_retrain=False)
    train_dataset, eval_dataset = data_manager.prepare_data(
        texts=df['text'].tolist(),
        labels=df['label'].values
    )

    # Train model
    model, history = train_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        vocab_size=CONFIG['vocab_size'],
        max_length=CONFIG['max_length'],
        strategy=strategy,
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        dropout_rate=CONFIG['dropout_rate'],
        epochs=CONFIG['epochs']
    )

    print("Training completed!")

if __name__ == "__main__":
    # Set memory growth for GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    main()