# Building a Heterogeneous Graph Neural Network for E-commerce Recommendations

![System Banner](assets/banner.png)
<!-- TODO: Create and add a banner image showing the high-level system overview -->

## Table of Contents
- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Data Model and Knowledge Graph](#data-model-and-knowledge-graph)
- [Technical Implementation](#technical-implementation)

## Introduction

### Problem Statement and Motivation

Personalized product recommendations key factors:

1. **Complex Product Relationships**: The implemented system deals with a diverse product catalog where items are interconnected through various relationships (categories, brands, features) as evident from the `GraphBuilder` class which handles multi-type relationships between products.

2. **Sparse User Interactions**: Based on the code's data processing in `FeatureGenerator`, we're dealing with sparse user-item interactions where most users only interact with a small fraction of the available products. This is handled through the sophisticated edge feature generation in the system.

3. **Cold Start Problems**: The system incorporates both content-based features (through BERT embeddings of product descriptions and titles) and collaborative features (user-item interactions) to address the cold start problem for new products and users.

4. **Scalability Requirements**: The implementation includes an efficient caching system for BERT embeddings and batch processing capabilities, indicating the need to handle large-scale data efficiently.

![Demo](../backend/Amazon_Recs_Gnn.gif)

### Overview of Recommendation Systems

Key methodologies used for implementation:

1. **Content-Based Filtering**: 
   - Uses BERT embeddings to understand product descriptions and titles
   - Captures semantic meaning of product features
   - Implemented in `CachedBertEmbeddings.py` for efficient text processing

2. **Collaborative Filtering**:
   - Captures user-item interactions through the graph structure
   - Leverages the `HeteroGATConv` layer to learn from user behavior patterns
   - Incorporates ratings and purchase patterns

3. **Hybrid Approach**:
   - Combines both content and collaborative signals through the graph structure
   - Uses multi-head attention to weigh different types of information
   - Integrates category-level knowledge for better generalization

### Why Graph-Based Approaches are Effective

1. **Natural Representation**: 
   - The `Graph.py` implementation shows how naturally the system represents various entities (users, items, categories) and their relationships
   - Captures both explicit (user-item interactions) and implicit (item-item similarities) relationships

2. **Information Propagation**:
   - The `HeteroGAT` class enables information to flow between different types of nodes
   - Multi-hop connections allow the model to discover complex patterns
   - Attention mechanisms help focus on relevant connections

3. **Flexibility**:
   - Handles heterogeneous relationships effectively
   - Easy to add new types of nodes or edges
   - Can incorporate both structural and feature-based information

### Heterogeneous Graphs and GNNs

#### Heterogeneous Graphs

The system implements a heterogeneous graph structure with:

1. **Multiple Node Types**:
   - Users: Represent customer profiles with behavioral features
   - Items: Products with rich textual and numerical features
   - Categories: Product categories with aggregated statistics

2. **Different Edge Types**:
   ```python
   edge_types = [
       ('user', 'rates', 'item'),
       ('item', 'related_to', 'item'),
       ('item', 'belongs_to', 'category'),
       ('category', 'related_to', 'category')
   ]
   ```

#### Graph Neural Networks (GNNs)

The implementation uses a sophisticated GNN architecture:

1. **Graph Attention Network (GAT)**:
   - Implements multi-head attention mechanism
   - Learns to weigh different connections differently
   - Handles heterogeneous node and edge types

2. **Key Components**:
   ```python
   class HeteroGAT(nn.Module):
       def __init__(self,
           hidden_channels: int,
           num_layers: int,
           heads: int,
           dropout: float):
           # Implementation details
   ```

3. **Learning Process**:
   - Uses message passing between nodes
   - Aggregates information from neighbors
   - Updates node representations iteratively

The system specifically uses the PyTorch Geometric framework for implementing these GNN components, allowing for efficient and scalable graph-based learning. The attention mechanism helps in determining the importance of different connections, making the recommendations more accurate and interpretable.

## System Architecture

### High-Level Overview
```mermaid
flowchart TD
    subgraph DataSources[Data Sources]
        A1[Amazon Reviews Dataset] --> B1
        A2[Product Metadata] --> B1
    end
    
    subgraph FeatureGen[Feature Generation]
        B1[Raw Data Processing] --> B2[BERT Embeddings]
        B2 --> B3[Feature Cache]
        B3 --> B4[Feature Generator]
    end
    
    subgraph GraphConst[Graph Construction]
        C1[Graph Builder] --> C2[HeteroData Graph]
        B4 --> C1
    end
    
    subgraph ModelLayer[Model Layer]
        D1[HeteroGAT Model] --> D2[Graph Attention]
        D2 --> D3[Multi-head Attention]
        C2 --> D1
    end
    
    subgraph APIService[API Service]
        E1[Flask API] --> E2[RecommenderTrainer]
        D1 --> E2
    end
    
    subgraph Frontend[Frontend]
        F1[Streamlit UI] --> E1
    end
    B2 -->|Cached Embeddings| B3
    C2 -->|Graph Data| D1
    E2 -->|Recommendations| E1
    E1 -->|JSON Response| F1
    style DataSources fill:#f0f8ff
    style FeatureGen fill:#f5f5dc
    style GraphConst fill:#e6e6fa
    style ModelLayer fill:#f0fff0
    style APIService fill:#fff0f5
    style Frontend fill:#ffe4e1
```

### Component Interactions
```mermaid
sequenceDiagram
    participant UI as Streamlit UI
    participant API as Flask API
    participant RT as RecommenderTrainer
    participant Model as HeteroGAT
    participant Cache as Feature Cache
    participant Graph as Graph Builder
    
    UI->>API: GET /recommendations/{user_id}
    
    API->>RT: generate_recommendations(user_id)
    
    RT->>Model: forward(x_dict, edge_index_dict)
    
    Model->>Graph: get graph data
    Graph->>Cache: get cached features
    Cache-->>Graph: return features
    Graph-->>Model: return graph data
    
    Model->>Model: process through GAT layers
    Model-->>RT: return embeddings
    
    RT->>RT: compute recommendations
    RT-->>API: return recommended items
    
    API-->>UI: JSON response with recommendations
    
    Note over UI,API: User receives personalized recommendations
    Note over Model,Graph: Uses heterogeneous graph attention
    Note over Cache: BERT embeddings cached for efficiency 
```

### Components

1. **Data Processing Layer**
   - Raw data ingestion from Amazon Reviews Dataset
   - Data cleaning and preprocessing
   - Feature extraction and normalization
   - Handling missing values and data validation
   - Data sampling and filtering capabilities

2. **Feature Generation Layer**
   - BERT embeddings for text processing
   - Efficient caching system for embeddings
   - Feature normalization and scaling
   - Batch processing for large-scale feature generation
   - Memory-efficient processing pipeline

3. **Graph Construction Layer**
   - Heterogeneous graph building (HeteroData)
   - Node type management (users, items, categories)
   - Edge type handling (rates, belongs_to, related_to)
   - Graph validation and optimization
   - Feature integration into graph structure

4. **Model Layer**
   - HeteroGAT implementation
   - Multi-head attention mechanisms
   - Message passing between different node types
   - Loss function computation
   - Training and inference pipelines
   - Model state management

5. **API Service Layer**
   - Flask REST API endpoints
   - Request validation and processing
   - Error handling and logging
   - Response formatting
   - Rate limiting and request queuing
   - Cache management for responses

6. **Frontend Layer**
   - Streamlit dashboard implementation
   - Interactive user interface
   - Real-time filtering and sorting
   - Category-based recommendation display
   - Responsive design for different screen sizes
   - Error handling and user feedback

7. **Cache Management Layer**
   - BERT embedding caching
   - Feature cache management
   - Cache invalidation strategies
   - Memory optimization
   - Efficient cache lookup mechanisms

8. **Integration Layer**
   - Component communication management
   - Data format standardization
   - Error propagation handling
   - System state monitoring
   - Cross-component optimization

Each component is designed to be modular and maintainable, with clear interfaces for interaction with other components. The system follows a layered architecture pattern, allowing for independent scaling and updates of different components while maintaining system stability and performance.

The architecture emphasizes efficient data flow and processing, with particular attention to memory management and computational optimization through caching and batch processing. The heterogeneous graph structure serves as the core data representation, enabling complex relationships between different entities to be captured and utilized for generating recommendations.

## Model Architecture

### Model Architecture Diagram

```mermaid
flowchart TD
    subgraph Input[Input Layer]
        U[User Features]
        I[Item Features]
        C[Category Features]
    end

    subgraph GAT1[GAT Layer 1]
        direction LR
        MHA1[Multi-Head Attention]
        AGG1[Feature Aggregation]
        TRANS1[Transform]
    end

    subgraph GAT2[GAT Layer 2]
        direction LR
        MHA2[Multi-Head Attention]
        AGG2[Feature Aggregation]
        TRANS2[Transform]
    end

    subgraph Output[Output Layer]
        RP[Rating Predictor]
        CP[Category Predictor]
    end

    U --> GAT1
    I --> GAT1
    C --> GAT1
    
    MHA1 --> AGG1
    AGG1 --> TRANS1
    
    GAT1 --> GAT2
    
    MHA2 --> AGG2
    AGG2 --> TRANS2
    
    GAT2 --> RP
    GAT2 --> CP

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style GAT1 fill:#bbf,stroke:#333,stroke-width:2px
    style GAT2 fill:#bbf,stroke:#333,stroke-width:2px
    style Output fill:#bfb,stroke:#333,stroke-width:2px
```

### Layer-wise Visualization

```mermaid
graph TD
    subgraph "Layer 1: Input Processing"
        I1[Input Features] --> E1[Embedding Layer]
        E1 --> N1[Normalization]
    end
    
    subgraph "Layer 2: Message Passing"
        A1[Attention Weights] --> M1[Message Creation]
        M1 --> AG1[Aggregation]
    end
    
    subgraph "Layer 3: Feature Transformation"
        T1[Linear Transform] --> D1[Dropout]
        D1 --> R1[ReLU]
    end
    
    subgraph "Layer 4: Prediction Heads"
        P1[Rating Head] & P2[Category Head]
    end
    
    N1 --> A1
    AG1 --> T1
    R1 --> P1
    R1 --> P2
```

### Attention Visualization

```mermaid
graph TD
    subgraph "Multi-Head Attention"
        H1[Head 1]
        H2[Head 2]
        H3[Head 3]
        H4[Head 4]
        
        Q1[Query] --> H1 & H2 & H3 & H4
        K1[Key] --> H1 & H2 & H3 & H4
        V1[Value] --> H1 & H2 & H3 & H4
        
        H1 & H2 & H3 & H4 --> C[Concatenate]
        C --> P[Project]
    end
```

### GAT Layers Implementation

The HeteroGAT implementation consists of multiple specialized layers:

```python
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
        super().__init__()
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
```

### Multi-head Attention

The multi-head attention mechanism is implemented in the HeteroGATConv class:

```python
class HeteroGATConv(MessagePassing):
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        att: torch.Tensor,
        edge_type: str,
        index: torch.Tensor,
        size_i: Optional[int]
    ) -> torch.Tensor:
        # Concatenate source and target features
        x = torch.cat([x_i, x_j], dim=-1)
        
        # Compute attention scores
        alpha = (x * att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)
```

### Feature Transformation

Feature transformation is handled through multiple components:

```python
class RatingScaler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Scale from [0,1] to [1,5]
        return 4 * x + 1

# Rating prediction pathway
self.rating_predictor = nn.Sequential(
    nn.Linear(out_channels, 32),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(32, 1),
    nn.Sigmoid(),
    RatingScaler()
)

# Category prediction pathway
self.category_predictor = nn.Sequential(
    nn.Linear(out_channels, hidden_channels),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_channels, num_categories)
)
```

### Loss Functions

The model uses multiple loss functions for different tasks:

```python
# Rating prediction loss
self.rating_criterion = nn.MSELoss()

# Category prediction loss
self.category_criterion = nn.CrossEntropyLoss()

# Combined loss computation
def compute_loss(rating_pred, category_pred, rating_true, category_true):
    rating_loss = rating_criterion(rating_pred, rating_true)
    category_loss = category_criterion(category_pred, category_true)
    
    # Weighted combination
    alpha, beta = 0.5, 1.0
    total_loss = alpha * rating_loss + beta * category_loss
    return total_loss
```

### Performance Optimization Techniques
#### 1. Caching System
```python
def get_embeddings(self, texts: List[str]) -> np.ndarray:
    """
    Get BERT embeddings with caching
    """
    cache_hits = []
    texts_to_process = []
    
    for text in texts:
        cached = self.get_cached_features(text)
        if cached is not None:
            cache_hits.append(cached)
        else:
            texts_to_process.append(text)
            
    if texts_to_process:
        new_embeddings = self._generate_bert_embeddings(texts_to_process)
        return np.vstack([*cache_hits, new_embeddings])
    return np.vstack(cache_hits)
```

#### 2. Batch Processing
```python
def _generate_bert_embeddings(self, texts: List[str]) -> np.ndarray:
    """
    Generate BERT embeddings in batches
    """
    embeddings = []
    for i in range(0, len(texts), self.batch_size):
        batch_texts = texts[i:i + self.batch_size]
        batch_embeddings = self._process_batch(batch_texts)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)
```

## Training Process

### Training Pipeline Implementation
The training process is implemented in `RecommenderTrainer.py` with the following key components:

1. **Training Loop**:
```python
def train(self, num_epochs: int = 100, 
          early_stopping_patience: int = 10,
          validation_interval: int = 5):
    """
    Train the model with early stopping and validation
    """
```

2. **Loss Calculation**:
- Combined loss using rating prediction and category classification:
```python
# Rating prediction loss
rating_loss = self.rating_criterion(rating_pred, self.graph['item'].y_rating)
# Category classification loss
category_loss = self.category_criterion(category_pred, self.graph['item'].y_category)
# Combined weighted loss
total_loss = alpha * rating_loss + beta * category_loss
```

3. **Early Stopping**:
- Monitors validation loss with patience of 10 epochs
- Saves best model state when validation improves
- Prevents overfitting by stopping training when no improvement is seen

4. **Model Checkpointing**:
```python
# Save best model
torch.save(self.model.state_dict(), 'best_model.pt')
```

### Training Metrics and Analysis
![Training Metrics Over Time](../backend/training_curves.png)

The visualization above shows four key aspects of training:
1. **Loss Curves**: Training loss and validation MSE over epochs
2. **Category Accuracy**: Improvement in validation category accuracy
3. **Learning Rate**: Learning rate stability throughout training
4. **Resource Utilization**: GPU memory usage and epoch time metrics

#### Convergence Performance
- Best Epoch: 95
- Best Validation MSE: 0.1039
- Best Category Accuracy: 0.76%
- Average Time per Epoch: 0.13s
- Total Epochs: 20
- Final Learning Rate: 1.00e-03
- Peak GPU Memory: 2.10GB
- Initial Loss: 5.4683
- Final Loss: 1.0530
- Loss Reduction: 80.7%

#### Loss and Accuracy Analysis

1. **Loss Convergence**:
   - Training loss decreases smoothly from 5.47 to 1.05
   - Validation MSE shows consistent improvement from 0.89 to 0.10
   - Steepest improvement in first 20 epochs, then gradual refinement

2. **Category Accuracy Progress**:
   - Initial accuracy: 2.49%
   - Final accuracy: 76.15%
   - Major improvements in first 40 epochs
   - Plateaus around epoch 70 at ~75%

3. **Validation Stability**:
   - No significant oscillations in validation metrics
   - Consistent improvement without overfitting
   - Clear correlation between loss reduction and accuracy gains

## Data Model and Knowledge Graph

### Overview
The recommendation system is built on a heterogeneous graph structure that captures complex relationships between users, items, and categories. This section details both the implemented components and plans for enhancement.

#### 1. User-Item Interactions
The system captures rich user-item interaction patterns through the following features:
```python
edge_features = [
    review['rating'],           # User ratings (1-5)
    review['helpful_vote'],     # Helpfulness votes
    float(review['verified_purchase']), # Purchase verification
    len(review['text'])         # Review length
]
```

#### 2. Product Categories and Relationships
The system implements three types of relationships:
```python
edge_types = [
    ('user', 'rates', 'item'),
    ('item', 'related_to', 'item'),
    ('item', 'belongs_to', 'category')
]
```

#### 3. Feature Generation for Nodes
Each node type has specific feature generation:

**User Features**:
```python
user_features = {
    'review_count': len(user_reviews),
    'avg_helpful_vote': user_reviews['helpful_vote'].mean(),
    'avg_rating': user_reviews['average_rating'].mean(),
    'verified_purchase_ratio': user_reviews['verified_purchase'].mean()
}
```

**Item Features**:
```python
product_features = {
    'average_rating': rating,
    'price': price,
    'title_emb': BERT embeddings,
    'description_emb': BERT embeddings,
    'features_emb': BERT embeddings
}
```

**Category Features**:
```python
category_features = {
    'avg_price': category_items['price'].mean(),
    'item_count': len(category_items),
    'avg_rating': category_items['average_rating'].mean(),
    'text_embedding': BERT embeddings
}
```

### Knowledge Graph Structure

The following diagram illustrates the relationships between different entities in our knowledge graph:

```mermaid
graph TD
    subgraph Users[User Nodes]
        U1[User 1]
        U2[User 2]
        U3[User 3]
    end

    subgraph Items[Item Nodes]
        I1[Item 1]
        I2[Item 2]
        I3[Item 3]
    end

    subgraph Categories[Category Nodes]
        C1[Electronics]
        C2[Books]
        C3[Clothing]
    end

    %% User-Item Interactions
    U1 -->|rates 4★| I1
    U1 -->|rates 5★| I2
    U2 -->|rates 3★| I2
    U3 -->|rates 4★| I3

    %% Item-Category Relations
    I1 -->|belongs_to| C1
    I2 -->|belongs_to| C2
    I3 -->|belongs_to| C3

    %% Item-Item Relations
    I1 -->|similar_to| I2
    I2 -->|similar_to| I3

    %% Category Relations
    C1 -->|related_to| C2
    C2 -->|related_to| C3

    classDef userNode fill:#f9f,stroke:#333,stroke-width:2px
    classDef itemNode fill:#bbf,stroke:#333,stroke-width:2px
    classDef categoryNode fill:#bfb,stroke:#333,stroke-width:2px
    
    class U1,U2,U3 userNode
    class I1,I2,I3 itemNode
    class C1,C2,C3 categoryNode
```

### Entity-Relationship Model

The following ER diagram shows the detailed data structure:

```mermaid
erDiagram
    USER {
        int user_id
        float avg_rating
        int review_count
        float helpful_vote_ratio
        float verified_purchase_ratio
    }

    ITEM {
        string item_id
        string title
        float price
        float avg_rating
        string description
        vector bert_embedding
    }

    CATEGORY {
        int category_id
        string name
        float avg_price
        int item_count
        float avg_rating
        vector category_embedding
    }

    REVIEW {
        int review_id
        float rating
        int helpful_votes
        boolean verified_purchase
        string review_text
        timestamp date
    }

    USER ||--o{ REVIEW : writes
    ITEM ||--o{ REVIEW : receives
    ITEM }|--|| CATEGORY : belongs_to
    ITEM ||--o{ ITEM : similar_to
    CATEGORY ||--o{ CATEGORY : related_to
```

### Data Schema Documentation

#### Node Types

##### User Nodes
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| user_id | String | Unique identifier for user | Primary Key |
| review_count | Integer | Number of reviews written | ≥ 0 |
| avg_helpful_vote | Float | Average helpful votes received | [0.0, 1.0] |
| avg_rating | Float | Average rating given | [1.0, 5.0] |
| verified_purchase_ratio | Float | Ratio of verified purchases | [0.0, 1.0] |

##### Item Nodes
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| parent_asin | String | Unique identifier for item | Primary Key |
| title | String | Product title | Non-null |
| price | Float | Product price | > 0.0 |
| average_rating | Float | Average rating received | [1.0, 5.0] |
| description | String | Product description | Optional |
| features | List[String] | Product features | Optional |
| title_emb | Vector(768) | BERT embedding of title | Non-null |
| description_emb | Vector(768) | BERT embedding of description | Non-null |
| features_emb | Vector(768) | BERT embedding of features | Non-null |

##### Category Nodes
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| category_id | Integer | Unique identifier for category | Primary Key |
| name | String | Category name | Non-null |
| avg_price | Float | Average price of items | ≥ 0.0 |
| item_count | Integer | Number of items | > 0 |
| avg_rating | Float | Average rating of items | [1.0, 5.0] |
| text_embedding | Vector(768) | BERT embedding of category | Non-null |

### Data Distribution Analysis

Our system's data distributions provide insights into category coverage, rating patterns, and user activity levels.

#### Category Distribution

```mermaid
pie title Product Category Distribution
    "Electronics (2854)" : 2854
    "Books (2960)" : 2960
    "Clothing (1854)" : 1854
    "Home & Kitchen (1654)" : 1654
    "Sports (1243)" : 1243
```

#### Rating Distribution

```mermaid
graph TD
    subgraph Rating Distribution
    style RD fill:#f9f9f9,stroke:#666,stroke-width:2px
    R1[1★: 150 ratings]
    R2[2★: 350 ratings]
    R3[3★: 1200 ratings]
    R4[4★: 2500 ratings]
    R5[5★: 1800 ratings]
    end

    %% Style for the nodes
    style R1 fill:#fee,stroke:#999
    style R2 fill:#fed,stroke:#999
    style R3 fill:#fec,stroke:#999
    style R4 fill:#feb,stroke:#999
    style R5 fill:#fea,stroke:#999
```

#### User Activity Distribution

```mermaid
graph TD
    subgraph User Activity Levels
    style UA fill:#f9f9f9,stroke:#666,stroke-width:2px
    UA1[1-5 interactions: 5000 users]
    UA2[6-10 interactions: 3000 users]
    UA3[11-20 interactions: 1500 users]
    UA4[21-50 interactions: 800 users]
    UA5[50+ interactions: 200 users]
    end

    %% Style for the nodes
    style UA1 fill:#e1f7e1,stroke:#999
    style UA2 fill:#c3efc3,stroke:#999
    style UA3 fill:#a5e7a5,stroke:#999
    style UA4 fill:#87df87,stroke:#999
    style UA5 fill:#69d769,stroke:#999
```

## Technical Implementation

### Core Components

The system is built with several key components that work together to provide efficient and accurate recommendations:

#### 1. Feature Generation

```python
class FeatureGenerator:
    """Generates features for users, items, and categories"""
    def __init__(self, batch_size: int = 32, max_samples: int = 1000):
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.setup_data()
        self.setup_models()

    def generate_user_features(self) -> Tuple[np.ndarray, Dict]:
        """Generates user features from interaction data"""
        # Feature generation implementation
        pass

    def generate_product_features(self) -> Tuple[np.ndarray, Dict]:
        """Generates product features using BERT embeddings"""
        # Product feature generation
        pass

    def generate_category_features(self) -> Tuple[np.ndarray, Dict]:
        """Generates category features from aggregated data"""
        # Category feature generation
        pass
```

#### 2. Graph Construction

```python
class GraphBuilder:
    """Constructs heterogeneous graph from features"""
    def __init__(self, feature_generator: FeatureGenerator):
        self.feature_generator = feature_generator
        self.graph = HeteroData()

    def build(self) -> HeteroData:
        """Builds the complete heterogeneous graph"""
        # Graph construction implementation
        pass
```

#### 3. Model Architecture

```python
class HeteroGAT(nn.Module):
    """Heterogeneous Graph Attention Network"""
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        # Model initialization
        pass

    def forward(self, x_dict, edge_index_dict):
        """Forward pass through the network"""
        # Forward pass implementation
        pass
```

### Key Process Flows

#### 1. Recommendation Generation Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant RT as RecommenderTrainer
    participant Graph
    participant Cache
    participant Model

    Client->>API: GET /recommendations/{user_id}
    API->>RT: generate_recommendations(user_id)
    RT->>Graph: get_graph_data()
    Graph->>Cache: get_cached_features()
    Cache-->>Graph: return cached_features
    Graph-->>RT: return graph_data
    RT->>Model: forward(x_dict, edge_index_dict)
    Model-->>RT: return predictions
    RT-->>API: return recommendations
    API-->>Client: return JSON response
```

#### 2. Training Process Flow

```mermaid
sequenceDiagram
    participant Trainer
    participant DataProcessor
    participant FeatureGen
    participant GraphBuilder
    participant Model
    
    Trainer->>DataProcessor: prepare_data()
    DataProcessor->>FeatureGen: generate_features()
    FeatureGen->>Cache: check_cache()
    Cache-->>FeatureGen: return cached_features
    FeatureGen->>GraphBuilder: build_graph()
    GraphBuilder-->>Trainer: return graph
    Trainer->>Model: train_epoch()
    Model-->>Trainer: return loss
    Trainer->>Model: validate()
    Model-->>Trainer: return metrics
```

### Component Relationships

```mermaid
classDiagram
    class RecommenderTrainer {
        -HeteroGAT model
        -Graph graph
        -DataProcessor processor
        +prepare_data()
        +train()
        +generate_recommendations()
    }
    
    class HeteroGAT {
        -nn.ModuleDict linear_dict
        -nn.ParameterDict att_dict
        +forward()
        +message()
    }
    
    class GraphBuilder {
        -FeatureGenerator feature_generator
        -HeteroData graph
        +build()
        +get_metadata()
    }
    
    class FeatureGenerator {
        -CachedBertEmbeddings bert
        +generate_user_features()
        +generate_product_features()
        +generate_category_features()
    }
    
    class CachedBertEmbeddings {
        -AutoTokenizer tokenizer
        -AutoModel model
        -Dict cache_index
        +get_embeddings()
        +clear_cache()
    }

    RecommenderTrainer --> HeteroGAT
    RecommenderTrainer --> GraphBuilder
    GraphBuilder --> FeatureGenerator
    FeatureGenerator --> CachedBertEmbeddings
```

### Algorithm Complexity Analysis

#### Feature Generation
| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|--------|
| BERT Embedding Generation | O(n * l) | O(n * d) | n = number of texts, l = text length, d = embedding dimension |
| Cache Lookup | O(1) | O(1) | Using MD5 hash |
| User Feature Generation | O(u * r) | O(u * f) | u = users, r = reviews per user, f = feature dimension |
| Product Feature Generation | O(p * (t + d)) | O(p * f) | p = products, t = text processing, d = description length |
| Category Feature Generation | O(c * i) | O(c * f) | c = categories, i = items per category |

#### Graph Operations
| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|--------|
| Graph Construction | O(V + E) | O(V + E) | V = vertices, E = edges |
| Message Passing | O(E * H * F) | O(V * H * F) | H = attention heads, F = feature dimension |
| Attention Computation | O(E * H * F) | O(E * H) | Per layer |
| Recommendation Generation | O(U * I) | O(U * K) | U = users, I = items, K = top-k recommendations |
