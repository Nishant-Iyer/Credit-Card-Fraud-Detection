import os
# Preemptively import xgboost before torch to avoid OpenMP threading segfaults on macOS
try:
    import xgboost
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AutoencoderNet(nn.Module):
    """
    PyTorch Autoencoder neural network architecture.
    """
    def __init__(self, input_dim: int, encoding_dim: int = 16):
        super(AutoencoderNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible wrapper for the PyTorch Autoencoder.
    Trains only on normal (non-fraudulent) transactions.
    Appends 'reconstruction_error' as a feature.
    """
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 16,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        model_path: str = "artifacts/autoencoder.pt",
        device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.device = device
        self.model = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AutoencoderFeatureExtractor":
        """
        Fits the Autoencoder only on normal transactions (where y == 0).
        """
        # Set device dynamically
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        elif self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"
            
        if self.device == "cpu":
            torch.set_num_threads(1)
            
        logger.info(f"Training Autoencoder on device: {self.device}")
        
        # Filter for normal transactions (Class = 0)
        if y is not None:
            normal_mask = (y == 0)
            X_normal = X[normal_mask].values.astype(np.float32)
        else:
            logger.warning("No labels provided to Autoencoder fit. Training on entire dataset.")
            X_normal = X.values.astype(np.float32)

        if len(X_normal) == 0:
            raise ValueError("No normal transactions found for Autoencoder training.")

        # Ensure correct input dimension
        actual_input_dim = X_normal.shape[1]
        self.model = AutoencoderNet(input_dim=actual_input_dim, encoding_dim=self.encoding_dim).to(self.device)
        
        # Prepare data loader
        tensor_x = torch.tensor(X_normal)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in loader:
                inputs = batch[0].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
            
            avg_loss = epoch_loss / len(X_normal)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Autoencoder Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")

        # Save model weights
        dir_name = os.path.dirname(self.model_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Autoencoder model saved to {self.model_path}")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates reconstruction error for all records and returns
        the original dataframe augmented with the 'reconstruction_error' column.
        """
        if self.device == "cpu":
            torch.set_num_threads(1)
            
        if self.model is None:
            # Try loading saved model
            actual_input_dim = X.shape[1]
            self.model = AutoencoderNet(input_dim=actual_input_dim, encoding_dim=self.encoding_dim).to(self.device)
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info(f"Loaded Autoencoder weights from {self.model_path}")
            else:
                raise RuntimeError("Autoencoder must be fitted before transforming.")
                
        self.model.eval()
        X_vals = X.values.astype(np.float32)
        tensor_x = torch.tensor(X_vals).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(tensor_x)
            # Calculate row-wise MSE
            reconstruction_error = torch.mean((tensor_x - reconstructed) ** 2, dim=1).cpu().numpy()
            
        X_out = X.copy()
        X_out["reconstruction_error"] = reconstruction_error
        return X_out
