import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List
import json
from pathlib import Path
import logging
from sampling_based_planning_framework.model.space_topology import MapCompletionNet

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MapCompletionPredictor:
    """
    A class for loading MapCompletionNet model and performing predictions.

    This class handles model loading from checkpoint, input preprocessing,
    prediction, and output conversion to numpy arrays.
    """

    def __init__(self,
                 checkpoint_path: str,
                 device: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the MapCompletionNet predictor.

        Args:
            checkpoint_path: Path to the model checkpoint file
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detection)
            config_path: Optional path to model configuration file
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = self._setup_device(device)
        self.model = None
        self.model_config = {}

        # Initialize model
        self._load_model()

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """
        Setup the device for model inference.

        Args:
            device: Preferred device string

        Returns:
            torch.device: The device to use for inference
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        device_obj = torch.device(device)
        logger.info(f"Using device: {device_obj}")

        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device_obj = torch.device('cpu')

        return device_obj

    def _load_model_config(self) -> Dict[str, Any]:
        """
        Load model configuration from file if provided.

        Returns:
            Dictionary containing model configuration
        """
        config = {}
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded model configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Could not load config from {self.config_path}: {e}")
        return config

    def _load_model(self):
        """
        Load the MapCompletionNet model from checkpoint.
        """
        try:
            # Load model configuration
            self.model_config = self._load_model_config()

            # Initialize model with config parameters or defaults
            init_channels = self.model_config.get('init_channels', 8)
            bottleneck_channels = self.model_config.get('bottleneck_channels', 32)

            self.model = MapCompletionNet(
                init_channels=init_channels,
                bottleneck_channels=bottleneck_channels
            )

            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            logger.info(f"Checkpoint loaded from {self.checkpoint_path}")

            # Handle different checkpoint formats
            state_dict = None
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint  # Assume direct state dict

            self.model.load_state_dict(state_dict)
            logger.info("Model parameters loaded successfully.")

            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model from {self.checkpoint_path}: {e}")

    def preprocess_input(self,
                         input_data: Union[np.ndarray, torch.Tensor, List],
                         input_size: tuple = (100, 100)) -> torch.Tensor:
        """
        Preprocess input data for model prediction.

        Args:
            input_data: Input data as numpy array, torch tensor, or list
            input_size: Expected input size (height, width)

        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert to tensor if not already
        if isinstance(input_data, list):
            input_data = np.array(input_data)

        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
        else:
            input_tensor = input_data.float()

        # Ensure correct shape and type
        if input_tensor.dim() == 2:
            # Single image: (H, W) -> (1, 1, H, W)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_tensor.dim() == 3:
            # Handle different 3D input formats
            if input_tensor.shape[0] == 1:
                # (1, H, W) -> (1, 1, H, W)
                input_tensor = input_tensor.unsqueeze(0)
            else:
                # (B, H, W) -> (B, 1, H, W)
                input_tensor = input_tensor.unsqueeze(1)

        # Resize if necessary
        if input_tensor.shape[2:] != input_size:
            logger.info(f"Resizing input from {input_tensor.shape[2:]} to {input_size}")
            input_tensor = torch.nn.functional.interpolate(
                input_tensor, size=input_size, mode='bilinear', align_corners=False
            )

        # Normalize to [0, 1] if not already
        if input_tensor.max() > 1.0 or input_tensor.min() < 0.0:
            input_tensor = torch.clamp(input_tensor, 0, 1)
            logger.warning("Input data was clamped to [0, 1] range")

        return input_tensor.to(self.device)

    def predict(self,
                input_data: Union[np.ndarray, torch.Tensor, List],
                return_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Perform prediction on input data.

        Args:
            input_data: Input data to predict on
            return_tensor: If True, return torch tensor instead of numpy array

        Returns:
            Prediction result as numpy array or torch tensor
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        try:
            # Preprocess input
            input_tensor = self.preprocess_input(input_data)

            # Perform prediction
            with torch.no_grad():
                prediction = self.model(input_tensor)

            # Convert to numpy array if requested
            if not return_tensor:
                prediction = prediction.cpu().numpy()

                # Remove batch and channel dimensions if single prediction
                if prediction.shape[0] == 1 and prediction.shape[1] == 1:
                    prediction = prediction[0, 0]  # (H, W)
                elif prediction.shape[0] == 1:
                    prediction = prediction[0]  # (C, H, W) or (H, W) if C=1

            return prediction

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_batch(self,
                      batch_data: Union[np.ndarray, torch.Tensor, List],
                      batch_size: int = 32,
                      return_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Perform batch prediction on multiple inputs.

        Args:
            batch_data: Batch of input data
            batch_size: Batch size for prediction
            return_tensor: If True, return torch tensor instead of numpy array

        Returns:
            Batch prediction results
        """
        # Preprocess entire batch
        batch_tensor = self.preprocess_input(batch_data)

        predictions = []

        # Process in batches to avoid memory issues
        for i in range(0, batch_tensor.shape[0], batch_size):
            batch = batch_tensor[i:i + batch_size]

            with torch.no_grad():
                batch_pred = self.model(batch)

            predictions.append(batch_pred)

        # Concatenate all predictions
        if return_tensor:
            result = torch.cat(predictions, dim=0)
        else:
            result = torch.cat(predictions, dim=0).cpu().numpy()

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "Model not loaded"}

        info = {
            "device": str(self.device),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_config": self.model_config,
            "input_size": "Adaptive (default 100x100)",
            "output_range": "[0, 1] (sigmoid activation)"
        }

        return info


# Example usage
if __name__ == "__main__":
    # Example usage
    try:
        # Initialize predictor
        predictor = MapCompletionPredictor(
            checkpoint_path="path/to/checkpoint.pth",
            config_path="path/to/config.json",  # Optional
            device="cuda"  # Optional, auto-detected
        )

        # Get model info
        print("Model Info:", predictor.get_model_info())

        # Create sample input (random data for demonstration)
        sample_input = np.random.rand(100, 100).astype(np.float32)

        # Perform prediction
        prediction = predictor.predict(sample_input)

        print(f"Input shape: {sample_input.shape}")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")

        # Batch prediction example
        batch_input = np.random.rand(4, 100, 100).astype(np.float32)  # 4 samples
        batch_prediction = predictor.predict_batch(batch_input)

        print(f"Batch prediction shape: {batch_prediction.shape}")

    except Exception as e:
        print(f"Error: {e}")
