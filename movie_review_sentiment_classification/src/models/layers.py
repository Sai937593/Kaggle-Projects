import tensorflow as tf
from tensorflow import keras
from keras import Layer, activations

class Positional_Embedding(Layer):
    def __init__(self, vocab_size, max_length, embed_dim, activation=None, **kwargs):
        super(Positional_Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.activation = activations.get(activation)
        
        # Create embedding layers
        self.token_embedding = keras.layers.Embedding(
            vocab_size, embed_dim,
            embeddings_initializer=keras.initializers.he_uniform(),
            name="token_embedding"
        )
        
        self.position_embedding = keras.layers.Embedding(
            max_length, embed_dim,
            embeddings_initializer=keras.initializers.he_uniform(),
            name="position_embedding"
        )

    def call(self, inputs, training=None):
        seq_length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_length, delta=1)
        
        token_embeddings = self.token_embedding(inputs)
        position_embeddings = self.position_embedding(positions)
        
        embeddings = token_embeddings + position_embeddings
        
        if self.activation is not None:
            embeddings = self.activation(embeddings)
            
        return embeddings

class Attention_Head(Layer):
    def __init__(self, head_dim, **kwargs):
        super(Attention_Head, self).__init__(**kwargs)
        self.head_dim = head_dim

    def build(self, input_shape):
        wt_shape = (input_shape[-1], self.head_dim)
        self.query_attention_weights = self.add_weight(
            shape=wt_shape,
            initializer=keras.initializers.he_uniform(),
            regularizer=keras.regularizers.l2(),
            trainable=True,
            name="Q_weights",
        )
        self.key_attention_weights = self.add_weight(
            shape=wt_shape,
            initializer=keras.initializers.he_uniform(),
            regularizer=keras.regularizers.l2(),
            trainable=True,
            name="K_weights",
        )
        self.value_attention_weights = self.add_weight(
            shape=wt_shape,
            initializer=keras.initializers.he_uniform(),
            regularizer=keras.regularizers.l2(),
            trainable=True,
            name="V_weights",
        )
        super().build(input_shape)

    def call(self, inputs):
        q = tf.matmul(inputs, self.query_attention_weights)
        k = tf.matmul(inputs, self.key_attention_weights)
        v = tf.matmul(inputs, self.value_attention_weights)

        qk = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(float(self.head_dim))
        soft_qk = keras.activations.softmax(qk)
        output = tf.matmul(soft_qk, v)
        return output

class Residual_Norm_Layer(Layer):
    def __init__(self, **kwargs):
        super(Residual_Norm_Layer, self).__init__(**kwargs)
        self.normlayer = keras.layers.LayerNormalization()
    
    def call(self, inputs):
        attention_output, original_input = inputs
        added = attention_output + original_input
        normalized = self.normlayer(added)
        return normalized