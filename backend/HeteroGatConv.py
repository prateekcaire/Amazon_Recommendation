import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Dict, List, Optional, Tuple, Union


class HeteroGATConv(MessagePassing):
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_types: List[Tuple[str, str, str]] = None
    ):
        super().__init__(aggr='add', node_dim=0)

        print(f"Initializing HeteroGATConv with edge types: {edge_types}")

        self.in_channels_dict = in_channels_dict
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_types = edge_types

        # Linear transformations for each node type
        self.linear_dict = nn.ModuleDict({
            node_type: nn.Linear(in_channels, out_channels * heads)
            for node_type, in_channels in in_channels_dict.items()
        })

        # Attention mechanisms for each edge type
        self.att_dict = nn.ParameterDict({
            f"{src}_to_{dst}": nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
            for src, _, dst in edge_types
        })

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize learnable parameters"""
        for linear in self.linear_dict.values():
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        for att in self.att_dict.values():
            nn.init.xavier_uniform_(att)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        print("\nInput shapes:")
        for k, v in x_dict.items():
            print(f"{k}: {v.shape}")

        # Transform node features and keep original dimensions
        transformed_features = {}
        for node_type, x in x_dict.items():
            # Linear transformation preserving original feature dimension
            transformed = self.linear_dict[node_type](x)
            # Reshape to (num_nodes, heads, out_channels)
            transformed = transformed.view(-1, self.heads, self.out_channels)
            transformed_features[node_type] = transformed
            print(f"Transformed {node_type} shape: {transformed.shape}")

        out_dict = {}
        for node_type, feat in transformed_features.items():
            # Keep the multi-head dimension initially
            out_dict[node_type] = feat
            print(f"Initial {node_type} output shape: {out_dict[node_type].shape}")

        # Process each edge type
        for edge_type in self.edge_types:
            src, edge_name, dst = edge_type
            if edge_type not in edge_index_dict:
                continue

            edge_index = edge_index_dict[edge_type]
            edge_type_key = f"{src}_to_{dst}"

            if edge_type_key not in self.att_dict:
                continue

            att = self.att_dict[edge_type_key]

            # message_output will have shape (num_dst_nodes, heads, out_channels)
            message_output = self.propagate(
                edge_index,
                x=(transformed_features[src], transformed_features[dst]),
                att=att,
                edge_type=edge_type_key
            )

            print(f"\nShapes for {edge_type}:")
            print(f"Source shape: {transformed_features[src].shape}")
            print(f"Target shape: {transformed_features[dst].shape}")
            print(f"Message output shape: {message_output.shape}")
            print(f"Current out_dict[{dst}] shape: {out_dict[dst].shape}")

            # Update destination node features
            # Both tensors should now have shape (num_dst_nodes, heads, out_channels)
            if dst in out_dict:
                out_dict[dst] = out_dict[dst] + message_output
            else:
                out_dict[dst] = message_output

        final_out_dict = {
            node_type: feat.mean(dim=1)
            for node_type, feat in out_dict.items()
        }

        print("\nFinal output shapes:")
        for k, v in final_out_dict.items():
            print(f"{k}: {v.shape}")

        return final_out_dict

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        att: torch.Tensor,
        edge_type: str,
        index: torch.Tensor,
        size_i: Optional[int]
    ) -> torch.Tensor:
        """
        Compute messages with attention.
        Input shapes:
            x_i: (num_edges, heads, out_channels)
            x_j: (num_edges, heads, out_channels)
            att: (1, heads, 2 * out_channels)
        Returns:
            tensor of shape (num_dst_nodes, heads, out_channels)
        """
        print(f"\nMessage function shapes for {edge_type}:")
        print(f"x_i: {x_i.shape}")
        print(f"x_j: {x_j.shape}")
        print(f"att: {att.shape}")

        # Concatenate source and target features
        x = torch.cat([x_i, x_j], dim=-1)  # Shape: (num_edges, heads, 2 * out_channels)

        # Compute attention scores
        alpha = (x * att).sum(dim=-1)  # Shape: (num_edges, heads)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, num_nodes=size_i)  # Shape: (num_edges, heads)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)  # Shape: (num_edges, heads, out_channels)


class RatingScaler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Scale from [0,1] to [1,5]
        return 4 * x + 1

class HeteroGAT(nn.Module):
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_categories: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        edge_types: List[Tuple[str, str, str]] = None
    ):
        """
        Multi-layer Heterogeneous Graph Attention Network

        Args:
            in_channels_dict: Dictionary of node types to their input feature dimensions
            hidden_channels: Number of hidden features
            out_channels: Number of output features
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
            edge_types: List of (source_node_type, edge_type, target_node_type) tuples
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Create GAT layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            HeteroGATConv(
                in_channels_dict=in_channels_dict,
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout,
                edge_types=edge_types
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            hidden_channels_dict = {node_type: hidden_channels for node_type in in_channels_dict}
            self.convs.append(
                HeteroGATConv(
                    in_channels_dict=hidden_channels_dict,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    edge_types=edge_types
                )
            )

        # Output layer
        if num_layers > 1:
            hidden_channels_dict = {node_type: hidden_channels for node_type in in_channels_dict}
            self.convs.append(
                HeteroGATConv(
                    in_channels_dict=hidden_channels_dict,
                    out_channels=out_channels,
                    heads=1,
                    dropout=dropout,
                    edge_types=edge_types
                )
            )

        self.rating_predictor = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Bounds to [0,1]
            RatingScaler()  # Scale to [1,5]
        )

        self.category_predictor = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_categories)
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all GAT layers

        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type

        Returns:
            Dictionary of output embeddings for each node type
        """
        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, edge_index_dict)
            if i != self.num_layers - 1:
                x_dict = {key: F.elu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                          for key, x in x_dict.items()}

        # Add category predictions to output
        x_dict['category_predictions'] = self.category_predictor(x_dict['item'])

        return x_dict