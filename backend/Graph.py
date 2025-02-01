import pandas as pd
import torch
import numpy as np
from typing import Dict, Any, List
from torch_geometric.data import HeteroData
from GraphFeatures import FeatureGenerator


class GraphBuilder:
    def __init__(self, feature_generator: FeatureGenerator):
        self.feature_generator = feature_generator
        self.graph = HeteroData()

    def _encode_categories(self, categories: pd.Series) -> List[int]:
        """
        Convert category labels to numeric indices using feature generator's mapping

        Args:
            categories: Series of category labels

        Returns:
            List of category indices
        """
        # Get category mapping from feature generator
        category_to_index = self.feature_generator.category_to_index

        # Convert categories to indices, using 0 for unknown categories
        encoded = [category_to_index.get(cat, 0) for cat in categories]

        return encoded

    def build(self) -> HeteroData:
        """
        Build heterogeneous graph using features from FeatureGenerator
        Returns:
            HeteroData: PyTorch Geometric heterogeneous graph
        """
        # Generate all features
        user_features, user_mapping = self.feature_generator.generate_user_features()
        product_features, product_mapping = self.feature_generator.generate_product_features()
        category_features, category_mapping = self.feature_generator.generate_category_features()
        # Create reverse mapping (index to category name)
        index_to_category = {idx: category for category, idx in category_mapping.items()}
        self.graph.index_to_category = index_to_category

        edge_features = self.feature_generator.generate_edge_features()
        edge_connectivity = self.feature_generator.generate_edge_connectivity()
        category_connectivity = self.feature_generator.generate_category_connectivity()

        target_labels = self.feature_generator.generate_target_labels()

        # Store the mappings and metadata in the graph
        self.graph.meta_data = {
            'meta_dataset': self.feature_generator.meta_dataset,
            'user_mapping': user_mapping,
            'product_mapping': product_mapping,
            'category_mapping': category_mapping
        }

        # Convert features to PyTorch tensors
        self.graph['user'].x = torch.FloatTensor(user_features)
        self.graph['item'].x = torch.FloatTensor(product_features)
        self.graph['category'].x = torch.FloatTensor(category_features)

        # Add edge connectivity and features for user-item edges
        self.graph['user', 'rates', 'item'].edge_index = edge_connectivity['user_item']
        self.graph['user', 'rates', 'item'].edge_attr = torch.FloatTensor(edge_features)

        # Add edge connectivity for item-item edges
        self.graph['item', 'related_to', 'item'].edge_index = edge_connectivity['item_item']

        item_to_category_edges = self._create_item_category_edges(product_mapping, category_mapping)
        self.graph['item', 'belongs_to', 'category'].edge_index = item_to_category_edges

        self.graph['category', 'related_to', 'category'].edge_index = category_connectivity

        # Add target labels
        self.graph['item'].y_category = torch.LongTensor(self._encode_categories(target_labels['category']))
        self.graph['item'].y_rating = torch.FloatTensor(target_labels['rating'])

        # Add metadata
        self.graph.num_user_nodes = len(user_mapping)
        self.graph.num_item_nodes = len(product_mapping)
        self.graph.num_category_nodes = len(category_mapping)

        return self.graph

    def _create_item_category_edges(self, product_mapping: Dict, category_mapping: Dict) -> torch.Tensor:
        """
        Create edges between items and their categories
        Args:
            product_mapping: Dictionary mapping product IDs to indices
            category_mapping: Dictionary mapping category names to indices
        Returns:
            torch.Tensor: Edge indices for item-category connections
        """
        meta_df = pd.DataFrame(self.feature_generator.meta_dataset)
        sources = []
        targets = []

        for _, row in meta_df.iterrows():
            if row['parent_asin'] in product_mapping and row['main_category'] in category_mapping:
                sources.append(product_mapping[row['parent_asin']])
                targets.append(category_mapping[row['main_category']])

        return torch.tensor([sources, targets], dtype=torch.long)

    @staticmethod
    def get_metadata(graph: HeteroData) -> Dict[str, Any]:
        """
        Get metadata about the constructed graph
        Args:
            graph: Constructed heterogeneous graph
        Returns:
            Dict containing graph metadata
        """
        metadata = {
            'num_users': graph.num_user_nodes,
            'num_items': graph.num_item_nodes,
            'num_categories': graph.num_category_nodes,
            'num_user_features': graph['user'].x.shape[1],
            'num_item_features': graph['item'].x.shape[1],
            'num_category_features': graph['category'].x.shape[1],
            'num_edge_features': graph['user', 'rates', 'item'].edge_attr.shape[1],
            'num_user_item_edges': graph['user', 'rates', 'item'].edge_index.shape[1],
            'num_item_item_edges': graph['item', 'related_to', 'item'].edge_index.shape[1],
            'num_item_category_edges': graph['item', 'belongs_to', 'category'].edge_index.shape[1],
            'num_category_category_edges': graph['category', 'related_to', 'category'].edge_index.shape[1]
        }
        return metadata


def create_graph(batch_size: int = 32, max_samples: int = 1000) -> HeteroData:
    """
    Convenience function to create the graph in one step
    Args:
        batch_size: Batch size for feature generation
        max_samples: Maximum number of samples to use
    Returns:
        HeteroData: Constructed heterogeneous graph
    """
    feature_generator = FeatureGenerator(batch_size=batch_size, max_samples=max_samples)
    graph_builder = GraphBuilder(feature_generator)
    return graph_builder.build()
