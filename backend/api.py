# Updated api.py
import traceback

from flask import Flask, jsonify
from RecommenderTrainer import RecommenderTrainer
from data_processor import DataProcessor
import os

app = Flask(__name__)

# Initialize model and processor
trainer = RecommenderTrainer(
    hidden_channels=64,
    num_layers=2,
    heads=4,
    dropout=0.2
)

# Prepare data without training
trainer.prepare_data(batch_size=32, max_samples=10000)

# Load the trained model if it exists, otherwise train
model_path = 'best_model.pt'
if os.path.exists(model_path):
    trainer.load_model(model_path)
else:
    print("No saved model found. Training new model...")
    trainer.train(num_epochs=10)  # Train if no saved model exists

data_processor = DataProcessor(trainer)


@app.route('/api/recommendations/<int:user_id>')
def get_recommendations(user_id):
    try:
        print(f"Number of users in graph: {trainer.graph.num_user_nodes}")
        print(
            f"User IDs available: {list(trainer.graph.meta_data['user_mapping'].keys())[:10]}...")  # Print first 10 user IDs
        print(f"Requested user_id: {user_id}")

        if user_id >= trainer.graph.num_user_nodes:
            return jsonify({
                'error': f'User ID {user_id} not found. Valid range is 0 to {trainer.graph.num_user_nodes - 1}'
            }), 404

        data = data_processor.get_recommendations_with_metadata(user_id)
        return jsonify(data)
    except Exception as e:
        print(f"Full error: {str(e)}")  # Print full error details
        print(f"Stack trace:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Disable the reloader/second process
