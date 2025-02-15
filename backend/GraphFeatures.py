import hashlib
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple, List
import torch.nn.functional as F


class FeatureGenerator:
    def __init__(self, batch_size: int = 32, max_samples: int = 1000):
        # Original initialization
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cache_dir = "./embedding_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = None
        self.model = None
        self.scaler = StandardScaler()

        # Add mappings as class attributes
        self.user_to_index = {}
        self.product_to_index = {}
        self.category_to_index = {}

        # Initialize other attributes
        self.setup_data()
        self.category_embeddings = {}

    def setup_data(self):
        """Initialize datasets with caching of processed data"""
        try:
            # Define cache paths
            cache_dir = "./data_cache"
            processed_meta_path = os.path.join(cache_dir, f"processed_meta_{self.max_samples}.pkl")
            processed_review_path = os.path.join(cache_dir, f"processed_review_{self.max_samples}.pkl")

            # Create cache directory if it doesn't exist
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            # Try to load processed datasets from cache
            if os.path.exists(processed_meta_path) and os.path.exists(processed_review_path):
                print("Loading processed datasets from cache...")
                with open(processed_meta_path, 'rb') as f:
                    self.meta_dataset = pickle.load(f)
                with open(processed_review_path, 'rb') as f:
                    self.review_dataset = pickle.load(f)
                return

            print("Processed datasets not found in cache. Loading from HuggingFace...")

            # If not in cache, load from HuggingFace
            all_configs = get_dataset_config_names('McAuley-Lab/Amazon-Reviews-2023')
            meta_configs = [c for c in all_configs if c.startswith('raw_meta_')]
            review_configs = [c for c in all_configs if c.startswith('raw_review_')]

            review_datasets = []
            for c in review_configs[:6]:
                dataset = load_dataset(
                    'McAuley-Lab/Amazon-Reviews-2023',
                    c,
                    split='full',
                    trust_remote_code=True,
                    cache_dir=cache_dir
                ).select(range(self.max_samples))
                review_datasets.append(dataset)

            initial_review_data = concatenate_datasets(review_datasets)

            product_ids = set()
            for review in initial_review_data:
                product_ids.add(review['parent_asin'])
            print(f"Found {len(product_ids)} unique products in reviews")

            meta_datasets = []
            for c in meta_configs[:6]:
                dataset = load_dataset(
                    'McAuley-Lab/Amazon-Reviews-2023',
                    c,
                    split='full',
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                filtered_dataset = dataset.filter(lambda x: x['parent_asin'] in product_ids)
                if len(filtered_dataset) > self.max_samples:
                    filtered_dataset = filtered_dataset.select(range(self.max_samples))
                meta_datasets.append(filtered_dataset)
                print(f"Loaded {len(filtered_dataset)} products with reviews from {c}")

            meta_data = concatenate_datasets(meta_datasets)

            valid_products = set(meta_data['parent_asin'])
            print(f"Final count of products with both reviews and metadata: {len(valid_products)}")

            self.review_dataset = initial_review_data.filter(lambda x: x['parent_asin'] in valid_products)
            self.meta_dataset = meta_data

            final_review_products = set(self.review_dataset['parent_asin'])
            final_meta_products = set(self.meta_dataset['parent_asin'])
            print(f"Verification - Products in reviews: {len(final_review_products)}")
            print(f"Verification - Products in metadata: {len(final_meta_products)}")
            print(f"Verification - Products overlap: {len(final_review_products.intersection(final_meta_products))}")

            # Save processed datasets to cache
            print("Saving processed datasets to cache...")
            with open(processed_meta_path, 'wb') as f:
                pickle.dump(self.meta_dataset, f)
            with open(processed_review_path, 'wb') as f:
                pickle.dump(self.review_dataset, f)

        except Exception as e:
            raise RuntimeError(f"Failed to load datasets: {str(e)}")

    def setup_models(self):
        """Initialize BERT model and tokenizer with error handling"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
            self.scaler = StandardScaler()
        except Exception as e:
            raise RuntimeError(f"Failed to load BERT model: {str(e)}")

    def _init_bert_if_needed(self):
        """CHANGE: Lazy initialization of BERT models to save memory when using cache"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.model is None:
            self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)

    def _get_cache_path(self, text: str) -> str:
        """CHANGE: Generate cache file path for a given text"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.npy")

    def get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """CHANGE: Modified to use caching for BERT embeddings"""
        start_time = time.time()
        embeddings = []
        texts_to_process = []
        cache_indices = []

        for idx, text in enumerate(texts):
            cache_path = self._get_cache_path(text)
            if os.path.exists(cache_path):
                # Load from cache
                embedding = np.load(cache_path)
                embeddings.append(embedding)
            else:
                # Mark for processing
                texts_to_process.append(text)
                cache_indices.append(idx)

        if texts_to_process:
            self._init_bert_if_needed()

            new_embeddings = []
            for i in range(0, len(texts_to_process), self.batch_size):
                batch_texts = texts_to_process[i:i + self.batch_size]
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
                new_embeddings.append(batch_embeddings)

            new_embeddings = np.vstack(new_embeddings)
            for idx, (text, embedding) in enumerate(zip(texts_to_process, new_embeddings)):
                cache_path = self._get_cache_path(text)
                np.save(cache_path, embedding)
                embeddings.append(embedding)

        # Ensure embeddings are in the original order
        if cache_indices:
            final_embeddings = []
            next_new_idx = 0
            for i in range(len(texts)):
                if i in cache_indices:
                    final_embeddings.append(embeddings[len(embeddings) - len(cache_indices) + next_new_idx])
                    next_new_idx += 1
                else:
                    final_embeddings.append(embeddings[next_new_idx])
            embeddings = final_embeddings

        # print(f"BERT embedding generation/loading took: {time.time() - start_time:.2f} seconds")
        return np.vstack(embeddings)

    def clear_embedding_cache(self):
        """CHANGE: Add method to clear the embedding cache if needed"""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.npy'):
                os.remove(os.path.join(self.cache_dir, filename))
        print("Embedding cache cleared")

    def generate_user_features(self) -> Tuple[np.ndarray, Dict]:
        """Generate user features and mapping"""
        start_time = time.time()
        review_df = pd.DataFrame(self.review_dataset)

        # Print debug info
        print(f"Total unique users in reviews: {review_df['user_id'].nunique()}")
        print(f"Sample user IDs: {review_df['user_id'].unique()[:5]}")

        user_features = {}
        user_to_index = {}

        for i, (user_id, user_reviews) in enumerate(review_df.groupby('user_id')):
            user_to_index[user_id] = i
            user_features[user_id] = {
                'review_count': len(user_reviews),
                'avg_helpful_vote': user_reviews['helpful_vote'].mean(),
                'avg_rating': user_reviews['rating'].mean(),
                'verified_purchase_ratio': user_reviews['verified_purchase'].mean()
            }

        # Print final mapping info
        print(f"Number of users mapped: {len(user_to_index)}")
        print(f"Index range: 0 to {len(user_to_index) - 1}")

        feature_matrix = np.array([[
            feat['review_count'],
            feat['avg_helpful_vote'],
            feat['avg_rating'],
            feat['verified_purchase_ratio']
        ] for feat in user_features.values()])

        print(f"User feature generation took: {time.time() - start_time:.2f} seconds")
        return self.scaler.fit_transform(feature_matrix), user_to_index

    def generate_product_features(self) -> Tuple[np.ndarray, Dict]:
        """Generate product features and mapping"""
        start_time = time.time()
        meta_df = pd.DataFrame(self.meta_dataset)
        product_features = {}
        product_to_index = {}

        for i, (_, meta) in enumerate(meta_df.iterrows()):
            product_id = meta['parent_asin']
            product_to_index[product_id] = i

            if product_id not in product_features:
                # Handle price conversion safely
                try:
                    price = float(meta['price']) if meta['price'] and meta['price'] != 'None' else round(random.uniform(5.0, 20.0), 2)
                except (ValueError, TypeError):
                    price = round(random.uniform(5.0, 20.0), 2)

                # Handle rating conversion safely
                try:
                    rating = float(meta['average_rating']) if meta['average_rating'] and meta[
                        'average_rating'] != 'None' else 0.0
                except (ValueError, TypeError):
                    rating = 0.0

                product_features[product_id] = {
                    'average_rating': rating,
                    'price': price,
                    'title_emb': self.get_bert_embeddings([meta['title'] or '']),
                    'description_emb': self.get_bert_embeddings([' '.join(meta['description'] or [])]),
                    'features_emb': self.get_bert_embeddings([' '.join(meta['features'] or [])])
                }
        print(f"Product feature generation took: {time.time() - start_time:.2f} seconds")
        return self._combine_product_features(product_features), product_to_index

    def _combine_product_features(self, product_features: Dict) -> np.ndarray:
        """Combine different product features into a single matrix"""
        # Extract numerical features
        numerical_features = np.array([[
            feat['average_rating'],
            feat['price']
        ] for feat in product_features.values()])

        # Extract embeddings separately
        title_embeddings = np.vstack([feat['title_emb'] for feat in product_features.values()])
        description_embeddings = np.vstack([feat['description_emb'] for feat in product_features.values()])
        features_embeddings = np.vstack([feat['features_emb'] for feat in product_features.values()])

        # Scale numerical features
        numerical_scaled = self.scaler.fit_transform(numerical_features)

        # Combine all features horizontally
        combined_features = np.hstack([
            numerical_scaled,
            title_embeddings,
            description_embeddings,
            features_embeddings
        ])

        return combined_features

    def generate_category_features(self) -> Tuple[np.ndarray, Dict]:
        """
        Generate category node features and mapping
        Returns:
            Tuple[np.ndarray, Dict]: Category features and category-to-index mapping
        """
        start_time = time.time()
        meta_df = pd.DataFrame(self.meta_dataset)

        # Replace None with a default category or filter out
        meta_df['main_category'] = meta_df['main_category'].fillna('Uncategorized')
        # Alternative: meta_df = meta_df[meta_df['main_category'].notna()]

        category_features = {}
        category_to_index = {}

        # Handle price conversion for the entire dataframe
        meta_df['price'] = pd.to_numeric(meta_df['price'].replace(['None', ''], np.nan), errors='coerce')
        meta_df['average_rating'] = pd.to_numeric(meta_df['average_rating'].replace(['None', ''], np.nan),
                                                  errors='coerce')

        # Create embeddings for each unique category
        for idx, category in enumerate(meta_df['main_category'].unique()):
            category_to_index[category] = idx

            # Get all items in this category
            category_items = meta_df[meta_df['main_category'] == category]

            # Generate category features from aggregated item information
            category_features[category] = {
                'avg_price': category_items['price'].mean() or 0.0,  # Use 0.0 if mean is NaN
                'item_count': len(category_items),
                'avg_rating': category_items['average_rating'].mean() or 0.0,  # Use 0.0 if mean is NaN
                # Generate text embedding from category name and sample product titles
                'text_embedding': self.get_bert_embeddings([
                    f"{category} " + " ".join(category_items['title'].fillna('').sample(min(5, len(category_items))))
                ])[0]
            }

        self.category_to_index = category_to_index

        # Combine numerical and text features
        feature_matrix = np.array([
            np.concatenate([
                [feat['avg_price'], feat['item_count'], feat['avg_rating']],
                feat['text_embedding']
            ]) for feat in category_features.values()
        ])
        print(f"Category feature generation took: {time.time() - start_time:.2f} seconds")
        return self.scaler.fit_transform(feature_matrix), category_to_index

    def generate_category_connectivity(self) -> Dict[str, torch.Tensor]:
        """
        Generate edges between related categories based on item similarity and user behavior
        Returns:
            Dict[str, torch.Tensor]: Category-category edge connectivity
        """
        meta_df = pd.DataFrame(self.meta_dataset)
        review_df = pd.DataFrame(self.review_dataset)

        meta_df['main_category'] = meta_df['main_category'].fillna('Uncategorized')

        category_cooccurrence = np.zeros((len(self.category_to_index), len(self.category_to_index)))

        # Group reviews by user to find category co-occurrences
        for _, user_reviews in review_df.groupby('user_id'):
            user_categories = meta_df[meta_df['parent_asin'].isin(user_reviews['parent_asin'])][
                'main_category'].unique()
            for cat1 in user_categories:
                for cat2 in user_categories:
                    if cat1 != cat2:
                        idx1 = self.category_to_index[cat1]
                        idx2 = self.category_to_index[cat2]
                        category_cooccurrence[idx1, idx2] += 1

        # Create edges for categories with significant co-occurrence
        sources, targets = [], []
        threshold = np.mean(category_cooccurrence) + np.std(category_cooccurrence)
        for i in range(len(self.category_to_index)):
            for j in range(len(self.category_to_index)):
                if category_cooccurrence[i, j] > threshold:
                    sources.append(i)
                    targets.append(j)

        return torch.tensor([sources, targets], dtype=torch.long)

    def generate_edge_features(self) -> np.ndarray:
        """Generate edge features for user-item interactions"""
        review_df = pd.DataFrame(self.review_dataset)
        edge_features = []

        for _, review in review_df.iterrows():
            edge_features.append([
                review['rating'],
                review['helpful_vote'],
                float(review['verified_purchase']),
                len(review['text'])
            ])

        return np.array(edge_features)

    def generate_target_labels(self):
        meta_df = pd.DataFrame(self.meta_dataset)
        # Return both category and rating information
        return {
            'category': meta_df['main_category'],
            'rating': meta_df['average_rating'].fillna(0).values
        }

    def generate_edge_connectivity(self) -> Dict[str, torch.Tensor]:
        """Generate edge connectivity for both user-item and item-item edges"""
        review_df = pd.DataFrame(self.review_dataset)
        meta_df = pd.DataFrame(self.meta_dataset)

        # User-Item edges
        user_item_sources = []
        user_item_targets = []
        for _, review in review_df.iterrows():
            if review['user_id'] in self.user_to_index and review['parent_asin'] in self.product_to_index:
                user_item_sources.append(self.user_to_index[review['user_id']])
                user_item_targets.append(self.product_to_index[review['parent_asin']])

        # Item-Item edges (products with same brand or category)
        item_item_sources = []
        item_item_targets = []

        # Group by category to create edges between items in same category
        for category in meta_df['main_category'].unique():
            category_products = meta_df[meta_df['main_category'] == category]['parent_asin'].unique()
            category_products = [p for p in category_products if p in self.product_to_index]

            # Create edges between all products in same category
            for i in range(len(category_products)):
                for j in range(i + 1, len(category_products)):
                    item_item_sources.append(self.product_to_index[category_products[i]])
                    item_item_targets.append(self.product_to_index[category_products[j]])

                    # Add reverse edge for undirected graph
                    item_item_sources.append(self.product_to_index[category_products[j]])
                    item_item_targets.append(self.product_to_index[category_products[i]])

        # Convert to tensors and ensure we have at least one edge
        if not user_item_sources:
            user_item_sources = [0]
            user_item_targets = [0]
        if not item_item_sources:
            item_item_sources = [0]
            item_item_targets = [0]

        return {
            'user_item': torch.tensor([user_item_sources, user_item_targets], dtype=torch.long),
            'item_item': torch.tensor([item_item_sources, item_item_targets], dtype=torch.long)
        }
