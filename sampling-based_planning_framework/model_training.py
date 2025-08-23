"""
Complete Training Function for MapCompletionNet with all required methods
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List
from tqdm import tqdm
from utils.utils import NumpyEncoder

from os.path import join, dirname, abspath
from scripts.data_loader import DataLoaderFactory
from model.space_topology import MapCompletionNet

class MapCompletionTrainer:
    def __init__(self, config_path: str):
        """
        Initialize trainer from config file

        Args:
            config_path: Path to config.yaml file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['training']['device'])

        # Initialize model
        self.model = MapCompletionNet(
            init_channels=self.config['model']['init_channels'],
            bottleneck_channels=self.config['model']['bottleneck_channels']
        )
        self.model.to(self.device)

        # Initialize trainer components
        self._setup_directories()
        self._setup_data_loaders()
        self._setup_optimizer_scheduler()
        self._setup_loss_function()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        print("Trainer initialized successfully!")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file with type conversion"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self._convert_config_types(config)
        return config

    def _convert_config_types(self, config: Dict[str, Any]):
        """Convert string values to appropriate types in config"""

        if 'training' in config:
            training_config = config['training']
            training_config['num_epochs'] = int(training_config.get('num_epochs', 100))
            training_config['batch_size'] = int(training_config.get('batch_size', 32))
            training_config['num_workers'] = int(training_config.get('num_workers', 4))

            if 'gradient_clip_value' in training_config:
                training_config['gradient_clip_value'] = float(training_config['gradient_clip_value'])

        if 'optimizer' in config:
            opt_config = config['optimizer']
            opt_config['lr'] = float(opt_config.get('lr', 0.001))
            opt_config['eps'] = float(opt_config.get('eps', 1e-8))
            opt_config['weight_decay'] = float(opt_config.get('weight_decay', 0.0001))

            if 'betas' in opt_config and isinstance(opt_config['betas'], list):
                opt_config['betas'] = [float(x) for x in opt_config['betas']]

        if 'scheduler' in config:
            sched_config = config['scheduler']
            sched_config['cosine_t_max'] = int(sched_config.get('cosine_t_max', 20))
            sched_config['eta_min'] = float(sched_config.get('eta_min', 1e-6))
            sched_config['plateau_factor'] = float(sched_config.get('plateau_factor', 0.5))
            sched_config['plateau_patience'] = int(sched_config.get('plateau_patience', 5))
            sched_config['min_lr'] = float(sched_config.get('min_lr', 1e-7))

    def _setup_directories(self):
        """Create necessary directories - 现在已定义"""
        self.assets_dir = Path(self.config['training']['assets_dir'])
        self.checkpoints_dir = Path(self.config['training']['checkpoints_dir'])

        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        print(f"Assets directory: {self.assets_dir}")
        print(f"Checkpoints directory: {self.checkpoints_dir}")

    def _setup_data_loaders(self):
        """Initialize data loaders from config"""
        data_config = self.config['data']

        self.train_loader, self.val_loader, self.test_loader = DataLoaderFactory.create_loaders(
            root_dir= join(dirname(abspath(__file__)), "datasets"),
            batch_size=self.config['training']["batch_size"],
            mask_ratio=self.config['training']["mask_ratio"],
            num_workers=4
        )

        print(f"Train samples: {len(self.train_loader)}")
        print(f"Validation samples: {len(self.val_loader)}")
        print(f"Test samples: {len(self.test_loader)}")

    def _setup_optimizer_scheduler(self):
        """Initialize optimizer and learning rate schedulers"""
        opt_config = self.config['optimizer']

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_config['lr'],
            betas=tuple(opt_config['betas']),
            eps=opt_config['eps'],
            weight_decay=opt_config['weight_decay']
        )

        # Schedulers
        sched_config = self.config['scheduler']
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=sched_config['cosine_t_max'] * len(self.train_loader),
            eta_min=sched_config['eta_min']
        )

        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=sched_config['plateau_factor'],
            patience=sched_config['plateau_patience'],
            min_lr=sched_config['min_lr']
        )

        print("Optimizer and schedulers initialized")

    def _setup_loss_function(self):
        """Initialize loss function"""
        loss_config = self.config['loss']
        self.criterion = nn.BCELoss()
        print("Loss function initialized: BCELoss")

    def train_epoch(self) -> float:
        """Train for one epoch and return average loss"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} Training")

        for batch_idx, (masked_inputs, original_targets) in enumerate(progress_bar):
            masked_inputs = masked_inputs.to(self.device)
            original_targets = original_targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(masked_inputs)
            loss = self.criterion(outputs, original_targets)

            # Backward pass
            loss.backward()

            # Gradient clipping if enabled
            if self.config['training'].get('gradient_clip', False):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_value']
                )

            self.optimizer.step()
            self.cosine_scheduler.step()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        return total_loss / num_batches

    def validate(self) -> float:
        """Validate model and return average loss"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} Validation")

            for masked_inputs, original_targets in progress_bar:
                masked_inputs = masked_inputs.to(self.device)
                original_targets = original_targets.to(self.device)

                outputs = self.model(masked_inputs)
                loss = self.criterion(outputs, original_targets)
                total_loss += loss.item()

                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def test(self) -> Dict[str, Any]:
        """Test model and return comprehensive results"""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for masked_inputs, original_targets in tqdm(self.test_loader, desc="Testing"):
                masked_inputs = masked_inputs.to(self.device)
                original_targets = original_targets.to(self.device)

                outputs = self.model(masked_inputs)
                loss = self.criterion(outputs, original_targets)
                total_loss += loss.item()

                all_outputs.append(outputs.cpu())
                all_targets.append(original_targets.cpu())

        # Calculate additional metrics
        outputs_tensor = torch.cat(all_outputs)
        targets_tensor = torch.cat(all_targets)

        # Binary accuracy
        predictions = (outputs_tensor > 0.5).float()
        accuracy = (predictions == targets_tensor).float().mean()

        # IoU (Intersection over Union)
        intersection = (predictions * targets_tensor).sum()
        union = (predictions + targets_tensor).clamp(0, 1).sum()
        iou = intersection / (union + 1e-8)

        return {
            'test_loss': total_loss / len(self.test_loader),
            'accuracy': accuracy.item(),
            'iou': iou.item(),
            'predictions': outputs_tensor,
            'targets': targets_tensor
        }

    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cosine_scheduler_state_dict': self.cosine_scheduler.state_dict(),
            'plateau_scheduler_state_dict': self.plateau_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoints_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoints_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cosine_scheduler.load_state_dict(checkpoint['cosine_scheduler_state_dict'])
        self.plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def save_training_logs(self):
        """Save training logs using custom encoder"""
        loss_data = {
            'epochs': list(range(1, len(self.train_losses) + 1)),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'timestamps': [datetime.now().isoformat() for _ in range(len(self.train_losses))]
        }

        with open(self.assets_dir / 'training_logs.json', 'w') as f:
            json.dump(loss_data, f, indent=2, cls=NumpyEncoder)

    def save_test_results(self, test_results: Dict[str, Any]):
        """Save test results using custom encoder"""
        test_summary = {
            'test_loss': test_results['test_loss'],
            'accuracy': test_results['accuracy'],
            'iou': test_results['iou'],
            'test_timestamp': datetime.now().isoformat(),
            'model_epoch': self.current_epoch
        }

        with open(self.assets_dir / 'test_results.json', 'w') as f:
            json.dump(test_summary, f, indent=2, cls=NumpyEncoder)

    def train(self, resume_checkpoint: str = None):
        """
        Main training function

        Args:
            resume_checkpoint: Path to checkpoint to resume training from (optional)
        """
        # Resume training if specified
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            self.load_checkpoint(resume_checkpoint)
            print(f"Resuming training from epoch {self.current_epoch}")

        start_time = time.time()
        num_epochs = self.config['training']['num_epochs']

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Step plateau scheduler
            self.plateau_scheduler.step(val_loss)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(is_best=is_best)

            # Save logs every epoch
            self.save_training_logs()

            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"LR: {current_lr:.2e}, Best Val: {self.best_val_loss:.4f}")

        # Training completed
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Final testing
        print("Running final test...")
        test_results = self.test()
        self.save_test_results(test_results)

        print(f"Test Results - Loss: {test_results['test_loss']:.4f}, "
              f"Accuracy: {test_results['accuracy']:.4f}, IoU: {test_results['iou']:.4f}")

        # Save final model and logs
        self.save_checkpoint()
        self.save_training_logs()


def main():
    """Main function to run training"""
    try:
        # Initialize trainer
        trainer = MapCompletionTrainer(join(dirname(abspath(__file__)), "config/config.yaml"))

        # Start training
        trainer.train(join(dirname(abspath(__file__)), "assets/best_model.pth"))

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()