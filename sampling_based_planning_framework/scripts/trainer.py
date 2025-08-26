"""
Optimizer and Training Configuration for MapCompletionNet
Designed for 15,000 samples and 100 epochs training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Any, Optional
import math



class MapCompletionTrainer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)

        # Training parameters
        self.batch_size = 32
        self.num_epochs = 100
        self.total_samples = 15000

        # Calculate training steps
        self.steps_per_epoch = math.ceil(self.total_samples / self.batch_size)
        self.total_steps = self.num_epochs * self.steps_per_epoch

        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._configure_optimizer()

        # Loss function
        self.criterion = nn.BCELoss()  # Binary Cross Entropy for binary segmentation

        # Track best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def _configure_optimizer(self) -> tuple:
        """
        Configure optimizer with carefully tuned parameters for map completion task

        Returns:
            tuple: (optimizer, scheduler)
        """
        # AdamW optimizer with weight decay for regularization
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,  # Initial learning rate
            betas=(0.9, 0.999),  # Adam beta parameters
            eps=1e-8,  # Epsilon for numerical stability
            weight_decay=1e-4,  # Weight decay for regularization
            amsgrad=False  # Don't use AMSGrad variant
        )

        # Combined learning rate scheduler
        # Cosine annealing with warm restarts + ReduceLROnPlateau
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=20 * self.steps_per_epoch,  # 20 epochs cycle
            eta_min=1e-6,  # Minimum learning rate
            last_epoch=-1
        )

        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # Reduce LR by half
            patience=5,  # Wait 5 epochs without improvement
            verbose=True,
            min_lr=1e-7  # Absolute minimum learning rate
        )

        return optimizer, {
            'cosine': cosine_scheduler,
            'plateau': plateau_scheduler
        }

    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

    def create_optimizer_checkpoint(self) -> Dict[str, Any]:
        """
        Create checkpoint dictionary for optimizer and scheduler state

        Returns:
            Dict: Checkpoint dictionary
        """
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cosine_scheduler_state_dict': self.scheduler['cosine'].state_dict(),
            'plateau_scheduler_state_dict': self.scheduler['plateau'].state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }

    def load_optimizer_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        Load optimizer and scheduler state from checkpoint

        Args:
            checkpoint: Dictionary containing optimizer and scheduler state
        """
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler['cosine'].load_state_dict(checkpoint['cosine_scheduler_state_dict'])
        self.scheduler['plateau'].load_state_dict(checkpoint['plateau_scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)

    def step_schedulers(self, val_loss: Optional[float] = None):
        """
        Step the learning rate schedulers

        Args:
            val_loss: Validation loss for plateau scheduler (optional)
        """
        # Step cosine scheduler (every batch)
        self.scheduler['cosine'].step()

        # Step plateau scheduler if validation loss is provided (every epoch)
        if val_loss is not None:
            self.scheduler['plateau'].step(val_loss)

    def get_training_plan(self) -> Dict[str, Any]:
        """
        Get detailed training plan with learning rate schedule

        Returns:
            Dict: Training plan details
        """
        return {
            'total_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'steps_per_epoch': self.steps_per_epoch,
            'total_steps': self.total_steps,
            'initial_lr': 1e-3,
            'min_lr': 1e-7,
            'weight_decay': 1e-4,
            'optimizer': 'AdamW',
            'schedulers': ['CosineAnnealingLR', 'ReduceLROnPlateau'],
            'loss_function': 'BCELoss',
            'expected_convergence': 'Around epoch 40-60',
            'recommended_checkpoints': [25, 50, 75, 100]
        }

    def print_training_summary(self):
        """Print training configuration summary"""
        plan = self.get_training_plan()

        print("=" * 60)
        print("MAP COMPLETION NET TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Total samples: {self.total_samples:,}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Total training steps: {self.total_steps:,}")
        print(f"Initial learning rate: {plan['initial_lr']:.1e}")
        print(f"Minimum learning rate: {plan['min_lr']:.1e}")
        print(f"Optimizer: {plan['optimizer']}")
        print(f"Weight decay: {plan['weight_decay']:.1e}")
        print(f"Expected convergence: {plan['expected_convergence']}")
        print("=" * 60)


# Additional utility functions
def create_gradient_clipping_hook(model: nn.Module, max_norm: float = 1.0):
    """
    Create gradient clipping hook to prevent exploding gradients

    Args:
        model: Model to apply gradient clipping to
        max_norm: Maximum gradient norm

    Returns:
        hook: Gradient clipping hook
    """

    def clip_gradients():
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    return clip_gradients


def calculate_effective_batch_size(batch_size: int, accumulation_steps: int = 1) -> int:
    """
    Calculate effective batch size for gradient accumulation

    Args:
        batch_size: Physical batch size
        accumulation_steps: Gradient accumulation steps

    Returns:
        int: Effective batch size
    """
    return batch_size * accumulation_steps


def get_recommended_batch_size(available_memory: float, model_params: int) -> int:
    """
    Get recommended batch size based on available memory

    Args:
        available_memory: Available GPU memory in GB
        model_params: Number of model parameters

    Returns:
        int: Recommended batch size
    """
    # Rough estimation: ~4 bytes per parameter for gradients + activations
    memory_per_sample = model_params * 4 * 2 / (1024 ** 3)  # in GB

    max_batch_size = int(available_memory * 0.8 / memory_per_sample)  # 80% of available memory
    return min(max_batch_size, 64)  # Cap at 64 for stability


# Example usage and training loop template
# def training_example():
#     """Example of how to use the training configuration"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Initialize model
#     model = MapCompletionNet(init_channels=8, bottleneck_channels=32)
#
#     train_loader, val_loader, test_loader = DataLoaderFactory.create_loaders(
#         root_dir=join(dirname(abspath(__file__)), "datasets"),
#         batch_size=32,
#         mask_ratio=0.9,
#         num_workers=4
#     )
#
#     # Initialize trainer
#     trainer = MapCompletionTrainer(model, device)
#     trainer.print_training_summary()
#
#     # Example training loop structure
#     for epoch in range(trainer.num_epochs):
#         # Training phase
#         model.train()
#         train_loss = 0.0
#
#         for batch_idx, (masked_inputs, original_targets) in enumerate(train_loader):
#             masked_inputs = masked_inputs.to(device)
#             original_targets = original_targets.to(device)
#
#             # Forward pass
#             outputs = model(masked_inputs)
#             loss = trainer.criterion(outputs, original_targets)
#
#             # Backward pass
#             trainer.optimizer.zero_grad()
#             loss.backward()
#
#             # Optional: Gradient clipping
#             # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#
#             trainer.optimizer.step()
#
#             # Step cosine scheduler (per batch)
#             trainer.step_schedulers()
#
#             train_loss += loss.item()
#
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for masked_inputs, original_targets in val_loader:
#                 masked_inputs = masked_inputs.to(device)
#                 original_targets = original_targets.to(device)
#
#                 outputs = model(masked_inputs)
#                 loss = trainer.criterion(outputs, original_targets)
#                 val_loss += loss.item()
#
#         # Step plateau scheduler (per epoch)
#         trainer.step_schedulers(val_loss)
#
#         # Save best model
#         if val_loss < trainer.best_val_loss:
#             trainer.best_val_loss = val_loss
#             trainer.best_epoch = epoch
#             # Save model checkpoint here
#
#         print(f'Epoch {epoch + 1}/{trainer.num_epochs}, '
#               f'LR: {trainer.get_learning_rate():.2e}, '
#               f'Train Loss: {train_loss / len(train_loader):.4f}, '
#               f'Val Loss: {val_loss / len(val_loader):.4f}')


# Pseudocode:
"""
CLASS MapCompletionTrainer:
    INIT(model, device):
        SET training parameters (batch_size=32, epochs=100)
        CALCULATE training steps
        CONFIGURE optimizer (AdamW, lr=1e-3, weight_decay=1e-4)
        CONFIGURE combined schedulers (CosineAnnealing + ReduceLROnPlateau)
        SET loss function (BCELoss)

    FUNCTION _configure_optimizer():
        USE AdamW with carefully tuned parameters
        COMBINE CosineAnnealingLR (for smooth decay) and ReduceLROnPlateau (for validation-based adjustment)
        RETURN optimizer and schedulers

    FUNCTION step_schedulers(val_loss):
        STEP cosine scheduler every batch
        STEP plateau scheduler every epoch using validation loss

    FUNCTION get_training_plan():
        RETURN detailed training configuration

UTILITY FUNCTIONS:
    create_gradient_clipping_hook(): Prevent exploding gradients
    calculate_effective_batch_size(): For gradient accumulation
    get_recommended_batch_size(): Memory-aware batch size calculation
"""

# if __name__ == "__main__":
#     training_example()