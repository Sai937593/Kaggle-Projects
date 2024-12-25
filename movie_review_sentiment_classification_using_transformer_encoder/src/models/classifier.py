import tensorflow as tf
from tensorflow import keras
from .encoder import Transformer_Encoder

class Transformer_Classifier(keras.Model):
    def __init__(
        self, 
        vocab_size, 
        max_length, 
        embed_dim,
        num_heads=8,
        num_layers=6,
        ff_dim=None,
        dropout_rate=0.1,
        **kwargs
    ):
        super(Transformer_Classifier, self).__init__(**kwargs)
        
        self.encoder = Transformer_Encoder(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        )
        
        self.classifier = keras.Sequential([
            keras.layers.GlobalAveragePooling1D(),
            
            # First dense block
            keras.layers.Dense(256,
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            
            # Second dense block
            keras.layers.Dense(128,
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            
            # Third dense block
            keras.layers.Dense(64,
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            
            # Output layer
            keras.layers.Dense(1,
                activation='sigmoid',
                kernel_initializer='he_normal')
        ])
        
    def call(self, inputs, training=None):
        encoder_output = self.encoder(inputs, training=training)
        return self.classifier(encoder_output, training=training)