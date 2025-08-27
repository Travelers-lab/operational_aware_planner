import json
import numpy as np
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import List, Union, Tuple, Any

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        # Let the base class default method raise the TypeError
        return super().default(obj)


class CoordinateConverter:
    """
    A class for converting coordinates between grid map space and workspace.

    This class handles bidirectional conversion between discrete grid coordinates
    and continuous workspace coordinates based on a specified resolution.

    Attributes:
        workspace_bounds (np.ndarray): 2D array defining workspace boundaries [[x_min, y_min], [x_max, y_max]]
        resolution (float): Grid resolution (units per cell)
        grid_width (int): Width of the grid in cells
        grid_height (int): Height of the grid in cells
    """

    def __init__(self, workspace_bounds: List[List[float]], resolution: float):
        """
        Initialize the coordinate converter with workspace boundaries and resolution.

        Args:
            workspace_bounds: List of two points defining workspace boundaries
                            [[min_x, min_y], [max_x, max_y]]
            resolution: Grid resolution (units per cell)

        Raises:
            ValueError: If workspace bounds are invalid or resolution is non-positive
        """
        if len(workspace_bounds) != 2 or len(workspace_bounds[0]) != 2 or len(workspace_bounds[1]) != 2:
            raise ValueError("Workspace bounds must contain exactly two 2D points")

        if resolution <= 0:
            raise ValueError("Resolution must be a positive number")

        self.workspace_bounds = np.array(workspace_bounds, dtype=float)
        self.resolution = float(resolution)

        # Calculate grid dimensions
        workspace_size = self.workspace_bounds[1] - self.workspace_bounds[0]
        self.grid_width = int(np.ceil(workspace_size[0] / self.resolution))
        self.grid_height = int(np.ceil(workspace_size[1] / self.resolution))

        # Adjust maximum bounds to fit exact grid multiples
        self.actual_max_bounds = self.workspace_bounds[0] + np.array(
            [self.grid_width, self.grid_height]) * self.resolution

    def grid_to_world(self, grid_coords: Union[List, np.ndarray, Tuple]) -> np.ndarray:
        """
        Convert grid coordinates to world coordinates.

        Args:
            grid_coords: Grid coordinates as (x, y) or array of coordinates

        Returns:
            np.ndarray: Corresponding world coordinates

        Raises:
            ValueError: If grid coordinates are outside valid range
        """
        grid_coords = np.array(grid_coords, dtype=float)

        if grid_coords.ndim == 1:
            # Single coordinate
            if not self._is_valid_grid_coord(grid_coords):
                raise ValueError(f"Grid coordinate {grid_coords} is out of bounds")

            world_coords = self.workspace_bounds[0] + grid_coords * self.resolution
            return world_coords

        else:
            # Multiple coordinates
            if not all(self._is_valid_grid_coord(coord) for coord in grid_coords):
                raise ValueError("One or more grid coordinates are out of bounds")

            world_coords = self.workspace_bounds[0] + grid_coords * self.resolution
            return world_coords

    def world_to_grid(self, world_coords: Union[List, np.ndarray, Tuple]) -> np.ndarray:
        """
        Convert world coordinates to grid coordinates.

        Args:
            world_coords: World coordinates as (x, y) or array of coordinates

        Returns:
            np.ndarray: Corresponding grid coordinates (floats, can be rounded to integers for discrete grid)

        Raises:
            ValueError: If world coordinates are outside workspace bounds
        """
        world_coords = np.array(world_coords, dtype=float)

        if world_coords.ndim == 1:
            # Single coordinate
            if not self._is_valid_world_coord(world_coords):
                raise ValueError(f"World coordinate {world_coords} is outside workspace bounds")

            grid_coords = (world_coords - self.workspace_bounds[0]) / self.resolution
            return grid_coords

        else:
            # Multiple coordinates
            if not all(self._is_valid_world_coord(coord) for coord in world_coords):
                raise ValueError("One or more world coordinates are outside workspace bounds")

            grid_coords = (world_coords - self.workspace_bounds[0]) / self.resolution
            return grid_coords

    def world_to_grid_discrete(self, world_coords: Union[List, np.ndarray, Tuple]) -> np.ndarray:
        """
        Convert world coordinates to discrete grid coordinates (integers).

        Args:
            world_coords: World coordinates as (x, y) or array of coordinates

        Returns:
            np.ndarray: Discrete grid coordinates as integers
        """
        grid_coords = self.world_to_grid(world_coords)
        return np.floor(grid_coords).astype(int)

    def _is_valid_grid_coord(self, grid_coord: np.ndarray) -> bool:
        """Check if grid coordinate is within valid range."""
        return (grid_coord >= 0).all() and (grid_coord[0] <= self.grid_width) and (grid_coord[1] <= self.grid_height)

    def _is_valid_world_coord(self, world_coord: np.ndarray) -> bool:
        """Check if world coordinate is within workspace bounds."""
        return (world_coord >= self.workspace_bounds[0]).all() and (world_coord <= self.actual_max_bounds).all()

    def get_grid_dimensions(self) -> Tuple[int, int]:
        """Return the dimensions of the grid (width, height)."""
        return self.grid_width, self.grid_height

    def get_workspace_bounds(self) -> np.ndarray:
        """Return the workspace boundaries."""
        return self.workspace_bounds.copy()