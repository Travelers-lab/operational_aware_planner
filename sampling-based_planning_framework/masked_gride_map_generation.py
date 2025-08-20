import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import random
import torch.nn.functional as F


class MapCompletionDataset(Dataset):
    """
    Dataset for map completion task with self-masking.
    Reads binary PNG images and applies masking to obstacle regions.
    """

    def __init__(self, data_dir, mask_ratio=0.9, edge_retention_ratio=0.25, transform=None):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Directory containing the PNG images
            mask_ratio (float): Ratio of obstacle pixels to mask (0-1)
            edge_retention_ratio (float): Ratio of edge pixels to retain (0-1)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.mask_ratio = mask_ratio
        self.edge_retention_ratio = edge_retention_ratio
        self.transform = transform

        # Get all PNG files in the directory
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        """Return the total number of samples"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: (masked_map, original_map) as torch tensors
        """
        # Load the image
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        # Convert to numpy array and ensure binary values (0 or 1)
        original_map = np.array(image) / 255.0
        original_map = (original_map > 0.5).astype(np.float32)

        # Create a copy for masking
        masked_map = original_map.copy()

        # Find obstacle regions (where value is 1)
        obstacle_indices = np.where(original_map == 1)

        if len(obstacle_indices[0]) > 0:
            # Detect edges of obstacles
            edge_mask = self.detect_edges(original_map)

            # Separate edge and non-edge obstacle pixels
            edge_obstacle_indices = np.where(edge_mask & (original_map == 1))
            non_edge_obstacle_indices = np.where(~edge_mask & (original_map == 1))

            # Calculate how many pixels to mask
            total_obstacle_pixels = len(obstacle_indices[0])
            pixels_to_mask = int(total_obstacle_pixels * self.mask_ratio)

            # Calculate how many edge pixels to retain
            total_edge_pixels = len(edge_obstacle_indices[0])
            edge_pixels_to_retain = int(total_edge_pixels * self.edge_retention_ratio)
            edge_pixels_to_mask = total_edge_pixels - edge_pixels_to_retain

            # Randomly select edge pixels to mask
            if edge_pixels_to_mask > 0:
                edge_mask_indices = random.sample(
                    range(total_edge_pixels),
                    edge_pixels_to_mask
                )
                for idx in edge_mask_indices:
                    i, j = edge_obstacle_indices[0][idx], edge_obstacle_indices[1][idx]
                    masked_map[i, j] = 0

            # Mask remaining pixels from non-edge obstacles
            non_edge_pixels_to_mask = pixels_to_mask - edge_pixels_to_mask
            if non_edge_pixels_to_mask > 0 and len(non_edge_obstacle_indices[0]) > 0:
                non_edge_mask_indices = random.sample(
                    range(len(non_edge_obstacle_indices[0])),
                    min(non_edge_pixels_to_mask, len(non_edge_obstacle_indices[0]))
                )
                for idx in non_edge_mask_indices:
                    i, j = non_edge_obstacle_indices[0][idx], non_edge_obstacle_indices[1][idx]
                    masked_map[i, j] = 0

        # Convert to PyTorch tensors
        masked_map = torch.from_numpy(masked_map).unsqueeze(0)  # Add channel dimension
        original_map = torch.from_numpy(original_map).unsqueeze(0)  # Add channel dimension

        # Apply transformations if any
        if self.transform:
            masked_map = self.transform(masked_map)
            original_map = self.transform(original_map)

        return masked_map, original_map

    def detect_edges(self, map_array):
        """
        Detect edges in the binary map using a simple convolution approach.

        Args:
            map_array (numpy.ndarray): Binary map array

        Returns:
            numpy.ndarray: Boolean array indicating edge pixels
        """
        # Convert to tensor for convolution operations
        map_tensor = torch.from_numpy(map_array).unsqueeze(0).unsqueeze(0).float()

        # Define edge detection kernel (Laplacian-like)
        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Apply convolution
        with torch.no_grad():
            edges = F.conv2d(map_tensor, kernel, padding=1)

        # Threshold to identify edges
        edges = (edges.abs() > 0.1).squeeze().numpy()

        return edges


def create_data_loaders(data_dir, batch_size=16, train_ratio=0.8, val_ratio=0.1,
                        mask_ratio=0.9, edge_retention_ratio=0.25):
    """
    Create data loaders for training, validation, and testing.

    Args:
        data_dir (str): Directory containing the PNG images
        batch_size (int): Batch size for data loaders
        train_ratio (float): Ratio of data to use for training
        val_ratio (float): Ratio of data to use for validation
        mask_ratio (float): Ratio of obstacle pixels to mask
        edge_retention_ratio (float): Ratio of edge pixels to retain

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, random_split

    # Create full dataset
    full_dataset = MapCompletionDataset(
        data_dir=data_dir,
        mask_ratio=mask_ratio,
        edge_retention_ratio=edge_retention_ratio
    )

    # Calculate dataset sizes
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# Example usage and test
if __name__ == "__main__":
    # Create a sample binary map for testing
    sample_map = np.zeros((100, 100), dtype=np.float32)

    # Add some obstacles
    sample_map[20:40, 20:40] = 1  # Square obstacle
    sample_map[60:80, 60:80] = 1  # Another square obstacle
    sample_map[30:50, 70:90] = 1  # Overlapping obstacle

    # Save as PNG
    os.makedirs("test_data", exist_ok=True)
    Image.fromarray((sample_map * 255).astype(np.uint8)).save("test_data/sample_map.png")

    # Test the dataset
    dataset = MapCompletionDataset("test_data", mask_ratio=0.9, edge_retention_ratio=0.25)
    masked, original = dataset[0]

    print(f"Original map shape: {original.shape}")
    print(f"Masked map shape: {masked.shape}")
    print(f"Original obstacle pixels: {torch.sum(original == 1).item()}")
    print(f"Masked obstacle pixels: {torch.sum(masked == 1).item()}")

    # Visualize the results (optional, requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original.squeeze(), cmap='gray')
        axes[0].set_title('Original Map')
        axes[0].axis('off')

        axes[1].imshow(masked.squeeze(), cmap='gray')
        axes[1].set_title('Masked Map')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig("test_data/masking_result.png")
        plt.close()

        print("Visualization saved to test_data/masking_result.png")
    except ImportError:
        print("Matplotlib not available, skipping visualization")