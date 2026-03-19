import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model.to(self.device)

    def train(self, X_train, y_train):
        epochs = self.config['training']['epochs']
        batch_size = self.config['training']['batch_size']
        learning_rate = self.config['training'].get('learning_rate', 0.001)

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        
        optimizer_name = self.config['model'].get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        history = {'loss': []}

        with mlflow.start_run():
            mlflow.log_params({
                "sequence_length": self.config['data']['sequence_length'],
                "train_split": self.config['data']['train_split'],
                "lstm_units": self.config['model']['lstm_units'],
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "optimizer": optimizer_name
            })
            
            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                # Setup tqdm progress bar for the epoch
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
                
                for batch_X, batch_y in pbar:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item()
                    epoch_loss += batch_loss * batch_X.size(0)
                    
                    # Update progress bar with current batch loss
                    pbar.set_postfix({'loss': f"{batch_loss:.6f}"})
                
                epoch_loss /= len(dataloader.dataset)
                history['loss'].append(epoch_loss)
                # print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}") # Handled by tqdm
                mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            
            mlflow.log_metric("final_train_loss", history['loss'][-1])
            return history

    def evaluate(self, X_test, y_test, scaler, processor):
        self.model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy()
            
        predictions = processor.inverse_transform(predictions)
        
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        
        print(f"Test RMSE: {rmse}")
        mlflow.log_metric("rmse", rmse)
        
        return {
            'rmse': rmse,
            'predictions': predictions
        }

    def save_model(self, export_path: str):
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        torch.save(self.model.state_dict(), export_path)
        print(f"Model saved to {export_path}")
