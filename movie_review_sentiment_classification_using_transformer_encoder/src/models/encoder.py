import tensorflow as tf
from tensorflow import keras
from .layers import Positional_Embedding, Attention_Head, Residual_Norm_Layer

class Transformer_Encoder(keras.Model):
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
        super(Transformer_Encoder, self).__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        if ff_dim is None:
            ff_dim = 4 * embed_dim
            
        # Initial embedding layer
        self.embedding = Positional_Embedding(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim
        )
        
        # Create tracked layers
        self.attention_heads = [[Attention_Head(head_dim=self.head_dim) 
                               for _ in range(num_heads)] 
                              for _ in range(num_layers)]
        
        self.attention_norms = [Residual_Norm_Layer() 
                              for _ in range(num_layers)]
        
        self.ff_layers = [keras.Sequential([
            keras.layers.Dense(ff_dim, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(embed_dim)
        ]) for _ in range(num_layers)]
        
        self.ff_norms = [Residual_Norm_Layer() 
                        for _ in range(num_layers)]
    
    def build(self, input_shape):
        super().build(input_shape)
            
    def call(self, inputs, training=None):
        x = self.embedding(inputs, training=training)
        
        for i in range(len(self.attention_heads)):
            attention_outputs = []
            for head in self.attention_heads[i]:
                attention_outputs.append(head(x))
            
            attention_outputs = tf.stack(attention_outputs, axis=-1)
            attention_output = tf.reshape(attention_outputs, 
                                        [-1, tf.shape(x)[1], self.embed_dim])
            
            x = self.attention_norms[i]([attention_output, x])
            ff_output = self.ff_layers[i](x, training=training)
            x = self.ff_norms[i]([ff_output, x])
            
        return x