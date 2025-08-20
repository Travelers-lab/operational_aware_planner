import os
import csv
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any
from omegaconf import OmegaConf


from model.space_topology import MapCompletionNet


@dataclass
class TrainingConfig:
    """Structured configuration schema for training parameters"""
    # Data parameters
    data_path: str = "data/"
    batch_size: int = 16
    num_workers: int = 4

    # Model parameters
    init_channels: int = 8
    bottleneck_channels: int = 32

    # Training parameters
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    checkpoint_dir: str = "checkpoints/"
    resume_checkpoint: Optional[str] = None

    # Optimization parameters
    optimizer: str = "adam"  # "adam" or "sgd"
    momentum: float = 0.9  # For SGD
    scheduler: str = "step"  # "step", "plateau", or "cosine"
    step_size: int = 30  # For step scheduler
    gamma: float = 0.1  # For step scheduler

    # Logging parameters
    log_interval: int = 10
    save_interval: int = 5


def train_model(config: TrainingConfig):
    """
    Train the MapCompletionNet model with the given configuration.
    Includes comprehensive logging of training, validation, and test metrics.

    Args:
        config: Training configuration parameters
    """
    # Create assets directory if it doesn't exist
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)

    # Create a unique experiment ID based on timestamp
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(assets_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create CSV file for logging metrics
    metrics_file = os.path.join(experiment_dir, "training_metrics.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'test_loss',
                         'learning_rate', 'epoch_time'])

    # Initialize TensorBoard writer
    tb_writer = SummaryWriter(log_dir=experiment_dir)

    # Initialize model
    model = MapCompletionNet(
        init_channels=config.init_channels,
        bottleneck_channels=config.bottleneck_channels
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load checkpoint if specified
    start_epoch = 0
    if config.resume_checkpoint:
        checkpoint = torch.load(config.resume_checkpoint, map_location=device)
        model.load_checkpoint(checkpoint)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from checkpoint: {config.resume_checkpoint}, starting at epoch {start_epoch}")

    # Define loss function
    criterion = nn.BCELoss()

    # Define optimizer
    if config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    # Load optimizer state if resuming
    if config.resume_checkpoint and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Learning rate scheduler
    if config.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )
    elif config.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    elif config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs
        )
    else:
        scheduler = None

    # TODO: Implement dataset - this is a placeholder
    # For a real implementation, you would load your actual dataset here
    # full_dataset = MapCompletionDataset(config.data_path)

    # Split dataset into train, validation, and test sets
    # For demonstration, we'll create dummy datasets
    # train_size = int(0.7 * len(full_dataset))
    # val_size = int(0.15 * len(full_dataset))
    # test_size = len(full_dataset) - train_size - val_size
    # train_dataset, val_dataset, test_dataset = random_split(
    #     full_dataset, [train_size, val_size, test_size]
    # )

    # Create data loaders
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=True,
    #     num_workers=config.num_workers
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=config.num_workers
    # )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=config.num_workers
    # )

    # For demonstration, we'll use placeholder data
    # In a real implementation, replace with the actual data loaders
    train_loader = [(torch.randn(config.batch_size, 1, 100, 100),
                     torch.randn(config.batch_size, 1, 100, 100)) for _ in range(100)]
    val_loader = [(torch.randn(config.batch_size, 1, 100, 100),
                   torch.randn(config.batch_size, 1, 100, 100)) for _ in range(20)]
    test_loader = [(torch.randn(config.batch_size, 1, 100, 100),
                    torch.randn(config.batch_size, 1, 100, 100)) for _ in range(20)]

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Log training progress
            if batch_idx % config.log_interval == 0:
                print(f'Epoch: {epoch + 1}/{config.epochs} '
                      f'Batch: {batch_idx}/{len(train_loader)} '
                      f'Loss: {loss.item():.6f}')

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)

        # Test phase (optional, can be done less frequently to save time)
        test_loss = 0.0
        if epoch % 5 == 0:  # Test every 5 epochs
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    test_loss += loss.item()

            # Calculate average test loss
            avg_test_loss = test_loss / len(test_loader)
        else:
            avg_test_loss = None

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if config.scheduler == "plateau":
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # Calculate epoch time
        epoch_time = time.time() - start_time

        # Log metrics to CSV
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                avg_test_loss if avg_test_loss is not None else '',
                current_lr,
                epoch_time
            ])

        # Log to TensorBoard
        tb_writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
        tb_writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
        if avg_test_loss is not None:
            tb_writer.add_scalar('Loss/Test', avg_test_loss, epoch + 1)
        tb_writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        # Print epoch summary
        print(f'Epoch {epoch + 1}/{config.epochs} - '
              f'Train Loss: {avg_train_loss:.6f}, '
              f'Val Loss: {avg_val_loss:.6f}, '
              f'{f"Test Loss: {avg_test_loss:.6f}, " if avg_test_loss is not None else ""}'
              f'LR: {current_lr:.6f}, '
              f'Time: {epoch_time:.2f}s')

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'params': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }

            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(experiment_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'params': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }, best_model_path)
            print(f"New best model saved with validation loss: {avg_val_loss:.6f}")

    # Final test after training completes
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.6f}")

    # Log final test results
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['final', '', '', avg_test_loss, '', ''])

    tb_writer.add_scalar('Loss/Final_Test', avg_test_loss)

    # Save final model
    final_model_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        'params': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'test_loss': avg_test_loss,
        'config': config
    }, final_model_path)

    tb_writer.close()
    print("Training completed.")


def main():
    """Main function to parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train Map Completion Network')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    args = parser.parse_args()

    # Load and validate configuration
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)

    # Start training
    train_model(cfg)

if __name__ == "__main__":
    main()