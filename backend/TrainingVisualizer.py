import matplotlib.pyplot as plt
import json
from typing import Dict
import os


class TrainingVisualizer:
    def __init__(self, metrics_file: str = 'training_metrics.json'):
        """Initialize visualizer with path to metrics file"""
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> Dict:
        """Load training metrics from JSON file"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}

    def plot_training_curves(self, save_path: str = 'training_curves.png'):
        """Create a comprehensive visualization of training metrics"""
        if not self.metrics:
            raise ValueError("No metrics data found")

        import seaborn as sns
        sns.set_style("whitegrid")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics Over Time', fontsize=16)

        # Plot 1: Loss Curves
        ax1.plot(self.metrics['epochs'], self.metrics['train_loss'], label='Training Loss')
        ax1.plot(self.metrics['epochs'], self.metrics['val_rating_mse'], label='Validation MSE')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot 2: Category Accuracy
        ax2.plot(self.metrics['epochs'], self.metrics['val_category_acc'],
                 color='green', label='Category Accuracy')
        ax2.set_title('Validation Category Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()

        # Plot 3: Learning Rate
        ax3.plot(self.metrics['epochs'], self.metrics['learning_rate'],
                 color='orange', label='Learning Rate')
        ax3.set_title('Learning Rate Over Time')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()

        # Plot 4: Resource Usage
        ax4.plot(self.metrics['epochs'], self.metrics['gpu_memory'],
                 color='red', label='GPU Memory')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(self.metrics['epochs'], self.metrics['epoch_time'],
                      color='purple', label='Epoch Time', linestyle='--')
        ax4.set_title('Resource Utilization')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('GPU Memory (bytes)')
        ax4_twin.set_ylabel('Time per Epoch (s)')
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def generate_training_report(self, save_path: str = 'training_report.md'):
        """Generate a markdown report of training metrics"""
        if not self.metrics:
            raise ValueError("No metrics data found")

        # Calculate summary statistics
        best_epoch = self.metrics['epochs'][
            self.metrics['val_rating_mse'].index(min(self.metrics['val_rating_mse']))
        ]
        best_mse = min(self.metrics['val_rating_mse'])
        best_acc = max(self.metrics['val_category_acc'])
        avg_epoch_time = sum(self.metrics['epoch_time']) / len(self.metrics['epoch_time'])

        report = f"""# Training Report

## Summary Statistics
- Best Epoch: {best_epoch}
- Best Validation MSE: {best_mse:.4f}
- Best Category Accuracy: {best_acc:.2f}%
- Average Time per Epoch: {avg_epoch_time:.2f}s

## Training Process
- Total Epochs: {len(self.metrics['epochs'])}
- Final Learning Rate: {self.metrics['learning_rate'][-1]:.2e}
- Peak GPU Memory: {max(self.metrics['gpu_memory']) / 1e9:.2f}GB

## Convergence Analysis
- Initial Loss: {self.metrics['train_loss'][0]:.4f}
- Final Loss: {self.metrics['train_loss'][-1]:.4f}
- Loss Reduction: {((self.metrics['train_loss'][0] - self.metrics['train_loss'][-1]) / self.metrics['train_loss'][0] * 100):.1f}%
"""

        with open(save_path, 'w') as f:
            f.write(report)


# Example usage:
if __name__ == "__main__":
    visualizer = TrainingVisualizer()
    visualizer.plot_training_curves()
    visualizer.generate_training_report()