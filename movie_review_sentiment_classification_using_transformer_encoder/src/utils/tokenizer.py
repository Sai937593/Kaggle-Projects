import os
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
import numpy as np

class TokenizerManager:
    def __init__(self, vocab_size=10000, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = None
        
    def load_or_create_tokenizer(self, force_retrain=False):
        """
        Load existing tokenizer or create new one if it doesn't exist
        """
        tokenizer_path = 'movie_review_tokenizer/vocab.txt'
        
        if os.path.exists(tokenizer_path) and not force_retrain:
            print("Loading existing tokenizer...")
            self.tokenizer = BertWordPieceTokenizer.from_file(tokenizer_path)
            return self.tokenizer
        
        print("Creating new tokenizer...")
        return self._train_new_tokenizer()
    
    def _train_new_tokenizer(self, texts=None):
        """
        Train a new tokenizer
        Args:
            texts: list of training texts
        """
        if texts is None:
            raise ValueError("texts must be provided for training new tokenizer")
            
        # Save training texts to file
        print("Preparing training data...")
        with open('train.txt', 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(str(text) + '\n')

        # Initialize and train tokenizer
        print("Training tokenizer...")
        self.tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=True,
            lowercase=True,
        )

        # Train the tokenizer
        self.tokenizer.train(
            files=['train.txt'],
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
            limit_alphabet=1000,
            wordpieces_prefix="##"
        )

        # Save the tokenizer
        print("Saving tokenizer...")
        os.makedirs('movie_review_tokenizer', exist_ok=True)
        self.tokenizer.save_model('movie_review_tokenizer')
        
        return self.tokenizer

    def encode_texts(self, texts):
        """
        Encode texts using the tokenizer
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call load_or_create_tokenizer first.")
            
        encodings = []
        attention_masks = []
        
        for text in tqdm(texts, desc="Encoding texts"):
            encoded = self.tokenizer.encode(text)
            
            input_ids = encoded.ids
            attention_mask = encoded.attention_mask
            
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            else:
                padding_length = self.max_length - len(input_ids)
                input_ids = input_ids + [self.tokenizer.token_to_id('[PAD]')] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                
            encodings.append(input_ids)
            attention_masks.append(attention_mask)
        
        return np.array(encodings), np.array(attention_masks)

    def check_coverage(self, encodings):
        """
        Check vocabulary coverage
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call load_or_create_tokenizer first.")
            
        unk_token_id = self.tokenizer.token_to_id('[UNK]')
        train_unks = (encodings == unk_token_id).sum()
        total_train_tokens = encodings.size
        
        coverage = (1 - train_unks/total_train_tokens) * 100
        
        print("\nVocabulary Coverage:")
        print(f"Unknown tokens: {train_unks}/{total_train_tokens}")
        print(f"Coverage: {coverage:.2f}%")
        
        return coverage