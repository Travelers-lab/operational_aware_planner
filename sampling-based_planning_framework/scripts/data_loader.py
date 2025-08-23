"""
Self-Masking Data Loader for Binary Grid Maps
This module provides data loading functions for self-supervised learning with masking.
"""
from torch.utils.data import Dataset, DataLoader
from os.path import join, dirname, abspath
import random
import os
from typing import List, Tuple
from PIL import Image
import numpy as np
import torch
from scipy import ndimage
import torchvision.transforms as transforms

class BinaryGridDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', obstacle_counts: List[int] = [1, 2, 3],
                 mask_ratio: float = 0.9, transform=None, preload: bool = True,
                 target_size: Tuple[int, int] = (100, 100)):
        """
        Initialize the binary grid dataset

        Args:
            root_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            obstacle_counts: List of obstacle count folders to include
            mask_ratio: Ratio of obstacle area to mask
            transform: Optional transforms to apply
            preload: Whether to preload all images into memory
            target_size: Target size for all images (height, width)
        """
        self.root_dir = root_dir
        self.split = split
        self.mask_ratio = mask_ratio
        self.transform = transform
        self.preload = preload
        self.target_size = target_size

        # Collect all image paths
        self.image_paths = []
        for count in obstacle_counts:
            count_dir = os.path.join(root_dir, split, str(count))
            if os.path.exists(count_dir):
                for img_name in os.listdir(count_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(count_dir, img_name))

        # Preload images if enabled
        self.preloaded_tensors = []
        if preload:
            print(f"Preloading {len(self.image_paths)} images...")
            for img_path in self.image_paths:
                tensor = self._load_and_preprocess_image(img_path)
                self.preloaded_tensors.append(tensor)
            print("Preloading complete.")

        print(f"Loaded {len(self.image_paths)} images from {split} split")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single data sample with masking

        Returns:
            masked_tensor: Tensor with masked obstacles (input)
            original_tensor: Original binary grid tensor (target)
        """
        # Load or get preloaded tensor
        if self.preload:
            original_tensor = self.preloaded_tensors[idx]
        else:
            img_path = self.image_paths[idx]
            original_tensor = self._load_and_preprocess_image(img_path)

        # Create masked version using optimized method
        masked_tensor = self._apply_obstacle_masking_fast(original_tensor.clone())

        # Ensure both tensors have the same shape
        assert masked_tensor.shape == original_tensor.shape, \
            f"Shape mismatch: masked {masked_tensor.shape} vs original {original_tensor.shape}"

        return masked_tensor, original_tensor

    def _load_and_preprocess_image(self, img_path: str) -> torch.Tensor:
        """
        Load PNG image and convert to binary tensor with consistent size
        """
        # Load and resize image
        img = Image.open(img_path).convert('L')

        # Resize to target size to ensure consistent dimensions
        resize_transform = transforms.Resize(self.target_size,
                                             interpolation=transforms.InterpolationMode.NEAREST)
        img = resize_transform(img)

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Invert if necessary using vectorized operation
        if np.mean(img_array) > 0.5:  # Mostly white background
            img_array = 1.0 - img_array

        # Convert to tensor and add channel dimension [1, H, W]
        tensor = torch.from_numpy(img_array).float().unsqueeze(0)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor

    def _apply_obstacle_masking_fast(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Fast obstacle masking using scipy's connected components
        """
        # Make a copy of the tensor
        masked_tensor = tensor.clone()

        # Convert to numpy for faster array operations
        tensor_np = tensor[0].numpy()  # Remove channel dim
        obstacle_mask = (tensor_np > 0.5).astype(np.uint8)

        if np.sum(obstacle_mask) == 0:
            return masked_tensor

        # Use scipy's connected components for faster region finding
        labeled_array, num_features = ndimage.label(obstacle_mask)

        # Get masked numpy array
        masked_np = masked_tensor[0].numpy()

        for label in range(1, num_features + 1):
            # Get region coordinates
            region_coords = np.argwhere(labeled_array == label)

            if len(region_coords) > 1:  # Only mask regions with multiple pixels
                self._mask_obstacle_region_fast(masked_np, region_coords)

        # Update the tensor with modified numpy array
        masked_tensor[0] = torch.from_numpy(masked_np).float()
        return masked_tensor

    def _mask_obstacle_region_fast(self, masked_array: np.ndarray, region_coords: np.ndarray):
        """
        Fast region masking using numpy operations
        """
        total_pixels = len(region_coords)
        mask_pixels = int(total_pixels * self.mask_ratio)

        if mask_pixels < 1:
            return

        H, W = masked_array.shape

        # Create a boolean mask for the region
        region_mask = np.zeros((H, W), dtype=bool)
        region_mask[region_coords[:, 0], region_coords[:, 1]] = True

        # Find interior pixels (pixels whose 4-connected neighbors are all in the region)
        interior_mask = np.ones_like(region_mask, dtype=bool)

        # Check 4-connected neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            shifted = np.roll(region_mask, shift=(dx, dy), axis=(0, 1))
            # Handle boundary conditions
            if dx == 1:
                shifted[-1, :] = False
            elif dx == -1:
                shifted[0, :] = False
            if dy == 1:
                shifted[:, -1] = False
            elif dy == -1:
                shifted[:, 0] = False

            interior_mask &= shifted

        # Get interior and edge pixels
        interior_coords = np.argwhere(interior_mask)
        edge_coords = np.argwhere(region_mask & ~interior_mask)

        # Convert to list of tuples for compatibility
        interior_pixels = [tuple(coord) for coord in interior_coords]
        edge_pixels = [tuple(coord) for coord in edge_coords]

        # Mask interior pixels first
        pixels_to_mask = interior_pixels

        # Add edge pixels if needed
        if len(pixels_to_mask) < mask_pixels and edge_pixels:
            additional_needed = mask_pixels - len(pixels_to_mask)
            if additional_needed > 0:
                # Randomly select edge pixels
                selected_indices = random.sample(range(len(edge_pixels)),
                                                 min(additional_needed, len(edge_pixels)))
                selected_edges = [edge_pixels[i] for i in selected_indices]
                pixels_to_mask.extend(selected_edges)

        # Apply masking
        for x, y in pixels_to_mask:
            masked_array[x, y] = 0.0

class DataLoaderFactory:
    """Factory class for creating data loaders for different splits"""

    @staticmethod
    def create_loaders(root_dir: str, batch_size: int = 32,
                       mask_ratio: float = 0.9, num_workers: int = 4):
        """
        Create data loaders for train, validation, and test splits

        Returns:
            train_loader, val_loader, test_loader
        """
        # Create datasets
        train_dataset = BinaryGridDataset(
            root_dir=root_dir,
            split='train',
            mask_ratio=mask_ratio
        )

        val_dataset = BinaryGridDataset(
            root_dir=root_dir,
            split='val',
            mask_ratio=mask_ratio
        )

        test_dataset = BinaryGridDataset(
            root_dir=root_dir,
            split='test',
            mask_ratio=mask_ratio
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader


# Utility functions
def visualize_sample(masked_tensor: torch.Tensor, original_tensor: torch.Tensor):
    """Visualize a sample pair"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Convert tensors to numpy for plotting
    masked_img = masked_tensor[0].numpy()  # Remove channel dim
    original_img = original_tensor[0].numpy()

    axes[0].imshow(masked_img, cmap='gray')
    axes[0].set_title('Masked Input')
    axes[0].axis('off')

    axes[1].imshow(original_img, cmap='gray')
    axes[1].set_title('Original Target')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def test_data_loading():
    """Test the data loading functionality"""
    # Create a small test dataset
    dataset = BinaryGridDataset(
        root_dir='binary_maps_dataset',  # Update with your actual path
        split='train',
        mask_ratio=0.9
    )

    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    if len(dataset) > 0:
        masked, original = dataset[0]
        print(f"Masked tensor shape: {masked.shape}")
        print(f"Original tensor shape: {original.shape}")
        print(f"Masked unique values: {torch.unique(masked)}")
        print(f"Original unique values: {torch.unique(original)}")

        # Visualize
        visualize_sample(masked, original)


# Pseudocode:
"""
CLASS BinaryGridDataset:
    INIT(root_dir, split, obstacle_counts, mask_ratio, transform):
        COLLECT all image paths from specified splits and obstacle counts
        STORE image paths and parameters

    FUNCTION __getitem__(idx):
        LOAD image from path
        CONVERT to binary tensor [1, H, W]
        APPLY obstacle masking with specified ratio
        PRESERVE obstacle edges during masking
        RETURN masked_tensor, original_tensor

    FUNCTION _apply_obstacle_masking(tensor):
        IDENTIFY obstacle regions using connected components
        FOR each obstacle region:
            CALCULATE number of pixels to mask based on mask_ratio
            IDENTIFY edge pixels (pixels with free-space neighbors)
            MASK interior pixels first
            IF needed, mask some edge pixels to reach target ratio
        RETURN masked tensor

    FUNCTION _find_obstacle_regions(mask):
        USE BFS to find connected components
        RETURN list of obstacle regions

CLASS DataLoaderFactory:
    STATIC METHOD create_loaders():
        CREATE datasets for train, val, test splits
        CREATE DataLoader instances with appropriate parameters
        RETURN train_loader, val_loader, test_loader
"""

# Example usage
if __name__ == "__main__":
    # Test the data loader
    test_data_loading()

    # Create full data loaders
    train_loader, val_loader, test_loader = DataLoaderFactory.create_loaders(
        root_dir=join(dirname(dirname(abspath(__file__))), "datasets"),
        batch_size=32,
        mask_ratio=0.9,
        num_workers=4
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Iterate through a batch
    for batch_idx, (masked, original) in enumerate(train_loader):
        print(f"Batch {batch_idx}: masked shape {masked.shape}, original shape {original.shape}")
        if batch_idx >= 2:  # Just check first few batches
            break