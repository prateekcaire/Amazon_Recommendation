# train.py
import torch

from RecommenderTrainer import RecommenderTrainer


def train_and_save_model():
    print("Checking available devices...")
    # Force CPU usage instead of MPS
    device = 'cpu'
    print("Using CPU for training")

    print("Initializing trainer...")
    trainer = RecommenderTrainer(
        hidden_channels=64,
        num_layers=2,
        heads=4,
        dropout=0.2,
        device=device
    )

    print("Preparing data...")
    trainer.prepare_data(batch_size=32, max_samples=1000)

    print("Training model...")
    trainer.train(num_epochs=100)  # Use full training regime
    print("Model training completed and saved to best_model.pt")


if __name__ == "__main__":
    train_and_save_model()
