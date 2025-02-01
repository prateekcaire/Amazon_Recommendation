import os
import pickle
import hashlib
from typing import List, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class CachedBertEmbeddings:
    def __init__(self, cache_dir: str = "./embedding_cache"):
        """
        Initialize the BERT embedding cache manager.

        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize BERT model and tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)

        # Load cache index if it exists
        self.cache_index_path = os.path.join(cache_dir, "cache_index.pkl")
        self.cache_index = self._load_cache_index()

    def _load_cache_index(self) -> Dict[str, str]:
        """Load the cache index from disk or create a new one."""
        if os.path.exists(self.cache_index_path):
            with open(self.cache_index_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_cache_index(self):
        """Save the cache index to disk."""
        with open(self.cache_index_path, 'wb') as f:
            pickle.dump(self.cache_index, f)

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the input text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cached embedding."""
        return os.path.join(self.cache_dir, f"{cache_key}.npy")

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Get BERT embeddings for a list of texts, using cache when available.

        Args:
            texts: List of texts to get embeddings for
            batch_size: Batch size for BERT processing

        Returns:
            numpy.ndarray: Array of embeddings
        """
        embeddings = []
        texts_to_process = []
        cache_keys_to_process = []

        # Check cache for each text
        for text in texts:
            cache_key = self._get_cache_key(text)
            cache_path = self._get_cache_path(cache_key)

            if os.path.exists(cache_path):
                # Load from cache
                embedding = np.load(cache_path)
                embeddings.append(embedding)
            else:
                # Mark for processing
                texts_to_process.append(text)
                cache_keys_to_process.append(cache_key)

        # Process uncached texts
        if texts_to_process:
            new_embeddings = self._generate_bert_embeddings(texts_to_process, batch_size)

            # Cache new embeddings
            for idx, (cache_key, embedding) in enumerate(zip(cache_keys_to_process, new_embeddings)):
                cache_path = self._get_cache_path(cache_key)
                np.save(cache_path, embedding)
                self.cache_index[cache_key] = texts_to_process[idx]
                embeddings.append(embedding)

            # Save updated cache index
            self._save_cache_index()

        return np.vstack(embeddings)

    def _generate_bert_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate BERT embeddings for uncached texts."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                max_length=128,
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def clear_cache(self):
        """Clear all cached embeddings."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.npy'):
                os.remove(os.path.join(self.cache_dir, filename))
        self.cache_index = {}
        self._save_cache_index()