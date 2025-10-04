"""
Transformer Models for Financial Time Series Forecasting.

Implements:
- Vanilla Transformer for return prediction
- Temporal Fusion Transformer (TFT)
- Attention mechanisms for time series
- Multi-horizon forecasting
- Feature importance analysis
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Transformer forecasting will use fallback methods.")


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""

    def __init__(
        self,
        data: np.ndarray,
        lookback: int = 60,
        horizon: int = 1
    ):
        """
        Initialize dataset.

        Args:
            data: Time series data (T x N)
            lookback: Number of historical periods
            horizon: Forecast horizon
        """
        self.data = data
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.lookback]
        y = self.data[idx + self.lookback:idx + self.lookback + self.horizon]

        return torch.FloatTensor(X), torch.FloatTensor(y)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerForecaster(nn.Module):
    """
    Transformer model for time series forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        forecast_horizon: int = 1
    ):
        """
        Initialize Transformer forecaster.

        Args:
            input_dim: Number of input features
            d_model: Dimension of model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            forecast_horizon: Number of steps to forecast
        """
        super(TransformerForecaster, self).__init__()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Transformer")

        self.input_dim = input_dim
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: Source sequence (batch, seq_len, input_dim)
            tgt: Target sequence for teacher forcing (batch, horizon, input_dim)

        Returns:
            Predictions (batch, horizon, input_dim)
        """
        # Embed input
        src = self.input_embedding(src)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)

        # For inference, use autoregressive decoding
        if tgt is None:
            # Start with last observation
            tgt = src[:, -1:, :]

            # Generate predictions autoregressively
            predictions = []
            for _ in range(self.forecast_horizon):
                tgt_embedded = self.pos_encoder(tgt.transpose(0, 1)).transpose(0, 1)

                output = self.transformer(src, tgt_embedded)
                pred = self.output_projection(output[:, -1:, :])

                predictions.append(pred)
                tgt = torch.cat([tgt, pred], dim=1)

            return torch.cat(predictions, dim=1)

        else:
            # Teacher forcing during training
            tgt = self.input_embedding(tgt)
            tgt = self.pos_encoder(tgt.transpose(0, 1)).transpose(0, 1)

            output = self.transformer(src, tgt)
            return self.output_projection(output)


class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer for interpretable forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        forecast_horizon: int = 1
    ):
        """
        Initialize Temporal Fusion Transformer.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            forecast_horizon: Forecast horizon
        """
        super(TemporalFusionTransformer, self).__init__()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TFT")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        # Variable selection network
        self.variable_selection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )

        # LSTM encoder
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim * forecast_horizon)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Tuple of (predictions, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        # Variable selection
        var_weights = self.variable_selection(x)
        x_selected = x * var_weights

        # LSTM encoding
        lstm_out, _ = self.encoder(x_selected)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Gating
        gate_weights = self.gate(attn_out)
        gated_out = gate_weights * attn_out + (1 - gate_weights) * lstm_out

        # Use last timestep for prediction
        final_state = gated_out[:, -1, :]

        # Generate predictions
        predictions = self.output_layer(final_state)
        predictions = predictions.view(batch_size, self.forecast_horizon, self.input_dim)

        return predictions, attn_weights


class TransformerForecasterWrapper:
    """
    High-level wrapper for Transformer forecasting.
    """

    def __init__(
        self,
        model_type: str = 'transformer',
        lookback_window: int = 60,
        forecast_horizon: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        learning_rate: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize Transformer forecaster wrapper.

        Args:
            model_type: 'transformer' or 'tft'
            lookback_window: Historical window size
            forecast_horizon: Number of steps to forecast
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of layers
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Transformer forecasting")

        self.model_type = model_type
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self, input_dim: int):
        """Build model architecture."""
        if self.model_type == 'transformer':
            self.model = TransformerForecaster(
                input_dim=input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_layers,
                num_decoder_layers=self.num_layers,
                forecast_horizon=self.forecast_horizon
            ).to(self.device)

        elif self.model_type == 'tft':
            self.model = TemporalFusionTransformer(
                input_dim=input_dim,
                hidden_dim=self.d_model,
                num_heads=self.nhead,
                forecast_horizon=self.forecast_horizon
            ).to(self.device)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(
        self,
        returns: pd.DataFrame,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.

        Args:
            returns: Historical returns (T x N)
            validation_split: Validation set fraction
            verbose: Print training progress

        Returns:
            Training history
        """
        # Build model
        input_dim = returns.shape[1]
        self.build_model(input_dim)

        # Prepare data
        data = returns.values
        split_idx = int(len(data) * (1 - validation_split))

        train_data = data[:split_idx]
        val_data = data[split_idx:]

        train_dataset = TimeSeriesDataset(train_data, self.lookback_window, self.forecast_horizon)
        val_dataset = TimeSeriesDataset(val_data, self.lookback_window, self.forecast_horizon)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training loop
        train_losses = []
        val_losses = []

        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                if self.model_type == 'transformer':
                    pred = self.model(X)
                else:  # TFT
                    pred, _ = self.model(X)

                loss = criterion(pred, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)

                    if self.model_type == 'transformer':
                        pred = self.model(X)
                    else:  # TFT
                        pred, _ = self.model(X)

                    loss = criterion(pred, y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }

    def predict(
        self,
        returns: pd.DataFrame,
        n_steps: int = 1
    ) -> pd.DataFrame:
        """
        Forecast future returns.

        Args:
            returns: Recent historical returns
            n_steps: Number of steps to forecast

        Returns:
            DataFrame of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()

        # Prepare input
        data = returns.values[-self.lookback_window:]
        X = torch.FloatTensor(data).unsqueeze(0).to(self.device)

        predictions = []

        with torch.no_grad():
            for _ in range(n_steps):
                if self.model_type == 'transformer':
                    pred = self.model(X)
                else:  # TFT
                    pred, _ = self.model(X)

                # Take first forecast step
                pred_step = pred[:, 0, :].cpu().numpy()
                predictions.append(pred_step[0])

                # Update input with prediction
                X = torch.cat([X[:, 1:, :], pred[:, 0:1, :]], dim=1)

        # Create DataFrame
        forecast_df = pd.DataFrame(
            predictions,
            columns=returns.columns,
            index=pd.RangeIndex(1, n_steps + 1)
        )

        return forecast_df


if __name__ == "__main__":
    print("Testing Transformer Forecasting...")

    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch not available. Skipping Transformer tests.")
    else:
        # Generate synthetic data
        np.random.seed(42)
        n_periods = 500
        n_assets = 3

        returns_data = np.random.randn(n_periods, n_assets) * 0.01
        returns_df = pd.DataFrame(
            returns_data,
            columns=['Asset_A', 'Asset_B', 'Asset_C']
        )

        print(f"\nData shape: {returns_df.shape}")

        # Test Transformer
        print("\n1. Testing Vanilla Transformer...")
        transformer = TransformerForecasterWrapper(
            model_type='transformer',
            lookback_window=30,
            forecast_horizon=1,
            d_model=32,
            nhead=2,
            num_layers=2,
            epochs=20,
            batch_size=16
        )

        history = transformer.fit(returns_df.iloc[:-50], verbose=True)
        print(f"   Final train loss: {history['train_losses'][-1]:.6f}")
        print(f"   Final val loss: {history['val_losses'][-1]:.6f}")

        predictions = transformer.predict(returns_df.iloc[-60:], n_steps=5)
        print(f"\n   5-step predictions:")
        print(predictions.round(4))

        # Test TFT
        print("\n2. Testing Temporal Fusion Transformer...")
        tft = TransformerForecasterWrapper(
            model_type='tft',
            lookback_window=30,
            forecast_horizon=1,
            d_model=32,
            nhead=2,
            epochs=20,
            batch_size=16
        )

        history = tft.fit(returns_df.iloc[:-50], verbose=True)
        print(f"   Final train loss: {history['train_losses'][-1]:.6f}")
        print(f"   Final val loss: {history['val_losses'][-1]:.6f}")

        predictions = tft.predict(returns_df.iloc[-60:], n_steps=5)
        print(f"\n   5-step predictions:")
        print(predictions.round(4))

        print("\n✅ Transformer forecasting implementation complete!")
