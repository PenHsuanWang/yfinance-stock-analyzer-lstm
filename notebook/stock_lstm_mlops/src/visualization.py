import matplotlib.pyplot as plt
import os
import pandas as pd

class TrainingVisualizer:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_learning_curve(self, history):
        if hasattr(history, 'history'):
            hist_dict = history.history
        else:
            hist_dict = history
            
        plt.figure(figsize=(10, 6))
        plt.plot(hist_dict['loss'], label='Train Loss', marker='o')
        if 'val_loss' in hist_dict:
            plt.plot(hist_dict['val_loss'], label='Validation Loss', marker='o')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        save_path = os.path.join(self.save_dir, 'learning_curve.png')
        plt.savefig(save_path)
        print(f"Learning curve saved to {save_path}")
        plt.close()

    def plot_predictions(self, data, training_data_len, predictions):
        train = data[:training_data_len].copy()
        valid = data[training_data_len:].copy()
        valid.loc[:, 'Predictions'] = predictions

        plt.figure(figsize=(16, 6))
        plt.title('Stock Price Prediction Model (PyTorch)')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        
        save_path = os.path.join(self.save_dir, 'predictions.png')
        plt.savefig(save_path)
        print(f"Predictions plot saved to {save_path}")
        plt.close()