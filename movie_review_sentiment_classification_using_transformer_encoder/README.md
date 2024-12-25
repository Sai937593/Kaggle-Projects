# Transformer-Based Sentiment Classifier

A custom implementation of a Transformer-based model for sentiment classification, built from scratch using TensorFlow/Keras. The implementation includes custom attention mechanisms, positional embeddings, and supports multi-GPU training.

## Features

- Custom implementation of Transformer architecture
- WordPiece tokenization with subword tokenization
- Multi-head self-attention mechanism
- Learned positional embeddings
- Multi-GPU support using TensorFlow's MirroredStrategy
- Comprehensive logging and visualization
- Residual connections and layer normalization
- Classification head with regularization

## Model Architecture

1. **Embedding Layer:**
   - WordPiece tokenization for efficient vocabulary usage
   - Learned positional embeddings to maintain sequence information
   - Dimension: configurable embedding size

2. **Transformer Encoder:**
   - Multi-head self-attention for parallel processing
   - Position-wise feed-forward networks
   - Layer normalization for training stability
   - Residual connections to prevent vanishing gradients

3. **Classification Head:**
   - Global average pooling for sequence aggregation
   - Three dense layers with batch normalization and dropout
   - L1L2 regularization for preventing overfitting
   - Final sigmoid activation for binary classification

