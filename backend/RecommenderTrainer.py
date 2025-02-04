import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm

from Graph import create_graph
from HeteroGatConv import HeteroGAT


class RecommenderTrainer:
    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 5e-4,
        device: str = None
    ):
        """
        Initialize the trainer for the recommender system

        Args:
            hidden_channels: Number of hidden features in GAT layers
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
            lr: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to run the model on (cuda/cpu)
        """
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.logger = logging.getLogger(__name__)
        # Force CPU usage instead of MPS
        self.device = 'cpu'
        print("Using CPU for training")

        self.logger.info(f"Using device: {self.device}")

        # Initialize logging
        logging.basicConfig(level=logging.INFO)

        self.graph = None
        self.model = None
        self.optimizer = None

    def prepare_data(self, batch_size: int = 32, max_samples: int = 1000):
        self.logger.info("Preparing graph data...")
        self.graph = create_graph(batch_size=batch_size, max_samples=max_samples)
        self.graph = self.graph.to(self.device)

        self.edge_types = [
            ('user', 'rates', 'item'),
            ('item', 'rated_by', 'user'),  # Added reverse edge
            ('item', 'related_to', 'item'),
            ('item', 'belongs_to', 'category'),
            ('category', 'related_to', 'category')
        ]

        # Create input channels dictionary
        self.in_channels_dict = {
            'user': self.graph['user'].x.shape[1],
            'item': self.graph['item'].x.shape[1],
            'category': self.graph['category'].x.shape[1]
        }

        self.logger.info(f"Edge types: {self.edge_types}")
        self.logger.info(f"Node feature dimensions: {self.in_channels_dict}")

    def setup_model(self):
        """Initialize the GAT model and optimizer"""
        self.logger.info("Setting up model...")

        # Get number of categories from graph metadata
        num_categories = self.graph.num_category_nodes

        self.model = HeteroGAT(
            in_channels_dict=self.in_channels_dict,
            hidden_channels=self.hidden_channels,
            out_channels=self.hidden_channels,  # Same as hidden for now
            num_categories=num_categories,  # Add this parameter
            num_layers=self.num_layers,
            heads=self.heads,
            dropout=self.dropout,
            edge_types=self.edge_types
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Define loss functions
        self.rating_criterion = nn.MSELoss()
        self.category_criterion = nn.CrossEntropyLoss()

        # Get and log model statistics
        self.get_model_stats()

    def get_model_stats(self) -> Dict[str, Any]:
        """Calculate model statistics and memory usage"""
        if self.model is None:
            self.logger.warning("Model not initialized. Call setup_model() first.")
            return {}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        layer_params = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                layer_params[name] = module.weight.numel()

        stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layer_parameters': layer_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'device': str(next(self.model.parameters()).device)
        }

        # Log the statistics
        self.logger.info(f"Model Statistics:")
        self.logger.info(f"Total Parameters: {stats['total_parameters']:,}")
        self.logger.info(f"Trainable Parameters: {stats['trainable_parameters']:,}")
        self.logger.info(f"Model Size: {stats['model_size_mb']:.2f} MB")
        self.logger.info(f"Device: {stats['device']}")

        return stats

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()

        self.logger.info(f"Available node types before forward pass: {list(self.graph.keys())}")
        self.logger.info(f"User features shape: {self.graph['user'].x.shape}")

        # Forward pass
        out_dict = self.model(
            x_dict={
                'user': self.graph['user'].x,
                'item': self.graph['item'].x,
                'category': self.graph['category'].x
            },
            edge_index_dict={
                ('user', 'rates', 'item'): self.graph['user', 'rates', 'item'].edge_index,
                ('item', 'rated_by', 'user'): self.graph['user', 'rates', 'item'].edge_index.flip(0),
                ('item', 'related_to', 'item'): self.graph['item', 'related_to', 'item'].edge_index,
                ('item', 'belongs_to', 'category'): self.graph['item', 'belongs_to', 'category'].edge_index,
                ('category', 'related_to', 'category'): self.graph['category', 'related_to', 'category'].edge_index
            }
        )

        self.logger.info(f"Node types after forward pass: {list(out_dict.keys())}")

        # Compute predictions
        item_embeddings = out_dict['item']
        rating_pred = self.model.rating_predictor(item_embeddings).squeeze(-1)  # Shape: (num_items,)
        category_pred = self.model.category_predictor(item_embeddings)

        # Calculate losses
        rating_loss = self.rating_criterion(rating_pred, self.graph['item'].y_rating)
        category_loss = self.category_criterion(category_pred, self.graph['item'].y_category)

        # Combined loss
        alpha, beta = 0.5, 1.0
        total_loss = alpha * rating_loss + beta * category_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return rating_loss.item(), category_loss.item()

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model using PyTorch's MSELoss for consistency
        """
        self.model.eval()

        with torch.no_grad():
            out_dict = self.model(
                x_dict={
                    'user': self.graph['user'].x,
                    'item': self.graph['item'].x,
                    'category': self.graph['category'].x
                },
                edge_index_dict={
                    ('user', 'rates', 'item'): self.graph['user', 'rates', 'item'].edge_index,
                    ('item', 'related_to', 'item'): self.graph['item', 'related_to', 'item'].edge_index,
                    ('item', 'belongs_to', 'category'): self.graph['item', 'belongs_to', 'category'].edge_index,
                    ('category', 'related_to', 'category'): self.graph['category', 'related_to', 'category'].edge_index
                }
            )

            item_embeddings = out_dict['item']
            rating_pred = self.model.rating_predictor(item_embeddings).squeeze(-1)
            category_pred = self.model.category_predictor(item_embeddings)

            # Use PyTorch's MSELoss for consistency with training
            rating_mse = self.rating_criterion(rating_pred, self.graph['item'].y_rating).item()

            category_accuracy = accuracy_score(
                self.graph['item'].y_category.cpu().numpy(),
                category_pred.argmax(dim=1).cpu().numpy()
            )

        return rating_mse, category_accuracy

    def train(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        validation_interval: int = 5
    ):
        """
        Train the model

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait before early stopping
            validation_interval: Number of epochs between validations
        """
        if self.model is None:
            self.setup_model()

        self.logger.info("Initial model statistics:")
        self.get_model_stats()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            rating_loss, category_loss = self.train_epoch()
            self.logger.info(
                f"Epoch {epoch}: "
                f"Rating Loss = {rating_loss:.4f}, "
                f"Category Loss = {category_loss:.4f}, "
            )

            # Validation
            if epoch % validation_interval == 0:
                val_rating_mse, val_category_acc = self.validate()
                val_loss = val_rating_mse - val_category_acc  # Combined metric

                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Rating Loss = {rating_loss:.4f}, "
                    f"Category Loss = {category_loss:.4f}, "
                    f"Val Rating MSE = {val_rating_mse:.4f}, "
                    f"Val Category Acc = {val_category_acc:.4f}"
                )

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info("Early stopping triggered!")
                        break

        self.logger.info("Final model statistics:")
        self.get_model_stats()

    def generate_recommendations(
        self,
        user_id: int,
        num_categories: int = 5,
        items_per_category: int = 40
    ) -> Dict[str, List[int]]:
        """
        Generate personalized recommendations for a user

        Args:
            user_id: User ID to generate recommendations for
            num_categories: Number of category rows to show
            items_per_category: Maximum items per category

        Returns:
            Dictionary mapping category names to lists of recommended item IDs
        """
        self.model.eval()
        with torch.no_grad():
            # Get embeddings
            out_dict = self.model(
                x_dict={
                    'user': self.graph['user'].x,
                    'item': self.graph['item'].x,
                    'category': self.graph['category'].x
                },
                edge_index_dict={
                    ('user', 'rates', 'item'): self.graph['user', 'rates', 'item'].edge_index,
                    ('item', 'related_to', 'item'): self.graph['item', 'related_to', 'item'].edge_index,
                    ('item', 'belongs_to', 'category'): self.graph['item', 'belongs_to', 'category'].edge_index,
                    ('category', 'related_to', 'category'): self.graph['category', 'related_to', 'category'].edge_index
                }
            )

            user_embedding = out_dict['user'][user_id]
            item_embeddings = out_dict['item']
            category_embeddings = out_dict['category']

            # Calculate user-category relevance scores
            category_scores = torch.matmul(category_embeddings, user_embedding)
            top_categories = torch.topk(category_scores, num_categories).indices

            recommendations = {}
            for category_idx in top_categories:
                # Get items in this category
                category_items = (self.graph['item', 'belongs_to', 'category'].edge_index[0]
                [self.graph['item', 'belongs_to', 'category'].edge_index[1] == category_idx])

                # Calculate item scores for this user and category
                category_item_embeddings = item_embeddings[category_items]
                item_scores = torch.matmul(category_item_embeddings, user_embedding)

                # Get top items
                top_items = torch.topk(item_scores, min(items_per_category, len(item_scores))).indices
                recommendations[str(category_idx.item())] = category_items[top_items].cpu().numpy().tolist()

            return recommendations

    def load_model(self, model_path: str = 'best_model.pt'):
        """
        Load a trained model from a saved state dict

        Args:
            model_path: Path to the saved model state dict
        """
        if self.model is None:
            self.setup_model()

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()  # Set to evaluation mode
            self.logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = RecommenderTrainer(
        hidden_channels=64,
        num_layers=2,
        heads=4,
        dropout=0.2
    )

    # Prepare data
    trainer.prepare_data(batch_size=32, max_samples=1000)

    # Train model
    trainer.train(num_epochs=100)

    # Generate recommendations for a user
    user_id = 0  # Example user
    recommendations = trainer.generate_recommendations(user_id)
    print("Recommendations:", recommendations)
