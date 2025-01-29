import random

import pandas as pd
from typing import Dict, Any
from RecommenderTrainer import RecommenderTrainer


class DataProcessor:
    def __init__(self, trainer: RecommenderTrainer):
        self.trainer = trainer
        self.meta_df = pd.DataFrame(trainer.graph.meta_data['meta_dataset'])
        self.category_mapping = self._get_inverse_mapping(trainer.graph.meta_data['category_mapping'])
        self.product_mapping = self._get_inverse_mapping(trainer.graph.meta_data['product_mapping'])

    @staticmethod
    def _get_inverse_mapping(mapping: Dict) -> Dict:
        """Create inverse mapping from index to original id"""
        return {v: k for k, v in mapping.items()}

    def get_metadata(self) -> Dict[str, Dict]:
        """Generate metadata for categories and products"""
        try:
            categories = {}
            products = {}

            # Process category metadata
            try:
                for idx, category in self.category_mapping.items():
                    try:
                        categories[str(idx)] = category
                    except Exception as e:
                        print(f"Error processing category {idx}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error in category processing: {str(e)}")
                categories = {}  # Fallback to empty categories

            # Process product metadata
            try:
                for idx, asin in self.product_mapping.items():
                    try:
                        # Find the product data safely
                        product_df = self.meta_df[self.meta_df['parent_asin'] == asin]
                        if product_df.empty:
                            print(f"No data found for product {asin}")
                            continue

                        product_data = product_df.iloc[0]

                        # Safely extract product information
                        products[str(idx)] = {
                            'title': product_data.get('title', 'No title available'),
                            'rating': float(product_data.get('average_rating', 0) if pd.notna(
                                product_data.get('average_rating')) else 0),
                            'price': float(product_data.get('price', round(random.uniform(5.0, 20.0), 2))
                                    if pd.notna(product_data.get('price')) and product_data.get('price') != 'None'
                                    else round(random.uniform(5.0, 20.0), 2)),
                            'image_url': product_data.get('images').get('large')[0]
                        }
                    except Exception as e:
                        print(f"Error processing product {asin}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error in product processing: {str(e)}")
                products = {}  # Fallback to empty products

            return {
                'categories': categories,
                'products': products
            }

        except Exception as e:
            print(f"Critical error in get_metadata: {str(e)}")
            # Return empty dictionaries if everything fails
            return {
                'categories': {},
                'products': {}
            }

    def get_recommendations_with_metadata(self, user_id: int) -> Dict[str, Any]:
        """Get recommendations and corresponding metadata"""
        # Get recommendations
        recommendations = self.trainer.generate_recommendations(user_id, num_categories=9)
        # Get metadata
        metadata = self.get_metadata()

        return {
            'recommendations': recommendations,
            'metadata': metadata
        }