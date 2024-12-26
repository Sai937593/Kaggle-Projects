# Transformer-Based Sentiment Classifier

A high-performing custom implementation of a Transformer-based model for sentiment classification, built from scratch using TensorFlow/Keras. Achieved **81% accuracy** on the Kaggle leaderboard, demonstrating strong performance in sentiment analysis tasks.

## Performance Highlights

- **Accuracy**: 81% on Kaggle leaderboard
- **Efficient Training**: Multi-GPU support enables faster training
- **Robust Generalization**: Advanced regularization techniques prevent overfitting

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

## Results

- Achieved 81% accuracy on Kaggle leaderboard
- Demonstrates robust performance across various sentiment analysis tasks
- Successfully handles complex sentence structures and nuanced expressions

