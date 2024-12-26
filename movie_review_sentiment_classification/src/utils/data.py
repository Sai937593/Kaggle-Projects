import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_and_split_data(texts, labels, test_size=0.2, random_state=42):
    """
    Split data into train and evaluation sets
    """
    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

def create_tf_dataset(encodings, labels, batch_size=32, is_training=True):
    """
    Create TensorFlow dataset with proper batching and prefetching
    """
    dataset = tf.data.Dataset.from_tensor_slices((encodings, labels))
    
    if is_training:
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

class DataManager:
    def __init__(self, tokenizer_manager, batch_size=32):
        self.tokenizer_manager = tokenizer_manager
        self.batch_size = batch_size
        
    def prepare_data(self, texts, labels, test_size=0.2):
        """
        Prepare data for training
        """
        # Split data
        train_texts, eval_texts, train_labels, eval_labels = load_and_split_data(
            texts, labels, test_size=test_size
        )
        
        # Encode texts
        train_encodings, train_masks = self.tokenizer_manager.encode_texts(train_texts)
        eval_encodings, eval_masks = self.tokenizer_manager.encode_texts(eval_texts)
        
        # Create datasets
        train_dataset = create_tf_dataset(
            train_encodings, 
            train_labels, 
            batch_size=self.batch_size,
            is_training=True
        )
        
        eval_dataset = create_tf_dataset(
            eval_encodings,
            eval_labels,
            batch_size=self.batch_size,
            is_training=False
        )
        
        return train_dataset, eval_dataset

    def save_encodings(self, train_encodings, train_masks, eval_encodings, eval_masks):
        """
        Save encodings to disk
        """
        os.makedirs('encoded_data', exist_ok=True)
        np.save('encoded_data/train_encodings.npy', train_encodings)
        np.save('encoded_data/train_masks.npy', train_masks)
        np.save('encoded_data/eval_encodings.npy', eval_encodings)
        np.save('encoded_data/eval_masks.npy', eval_masks)
        
    def load_encodings(self):
        """
        Load encodings from disk
        """
        return (
            np.load('encoded_data/train_encodings.npy'),
            np.load('encoded_data/train_masks.npy'),
            np.load('encoded_data/eval_encodings.npy'),
            np.load('encoded_data/eval_masks.npy')
        )