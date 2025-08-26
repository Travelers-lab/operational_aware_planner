import numpy as np


class BinaryMapGenerator:
    """
    Generates a binary occupancy grid map where 1 represents occupied cells
    and 0 represents free cells.

    Attributes:
        map_size (tuple): Dimensions of the grid map (height, width)
        grid (numpy.ndarray): The generated binary grid
    """

    def __init__(self, map_size=(100, 100)):
        """
        Initializes the binary map generator.

        Args:
            map_size (tuple, optional): Grid dimensions (height, width).
                Defaults to (100, 100).
        """
        self.map_height, self.map_width = map_size
        self.grid = np.zeros(map_size, dtype=np.uint8)

    def generate(self, coordinates):
        """
        Generates a binary occupancy grid from input coordinates.

        Args:
            coordinates: Input points as either:
                - List of (x, y) tuples
                - List of [x, y] lists
                - N×2 numpy array
                - Flattened list/array of coordinates [x1, y1, x2, y2, ...]

        Returns:
            numpy.ndarray: 100×100 binary grid with occupied cells marked as 1

        Raises:
            ValueError: For invalid input formats or out-of-bound coordinates
        """
        # Reset grid to all zeros
        self.grid.fill(0)

        # Convert input to standardized numpy array format
        points = self._validate_and_convert(coordinates)

        # Process each coordinate pair
        for x, y in points:
            # Convert to integer indices
            ix = int(round(x))
            iy = int(round(y))

            # Validate coordinate range
            if not (0 <= ix < self.map_width and 0 <= iy < self.map_height):
                raise ValueError(
                    f"Coordinate ({x}, {y}) is out of bounds. "
                    f"Valid range: x[0-{self.map_width - 1}], y[0-{self.map_height - 1}]"
                )

            # Mark cell as occupied
            self.grid[iy, ix] = 1

        return self.grid.copy()

    def _validate_and_convert(self, coordinates):
        """
        Validates input format and converts to standardized N×2 numpy array.

        Args:
            coordinates: Input points in various formats

        Returns:
            numpy.ndarray: N×2 array of coordinates

        Raises:
            ValueError: For unsupported or malformed input
        """
        if isinstance(coordinates, np.ndarray):
            # Handle numpy array input
            if coordinates.ndim == 2 and coordinates.shape[1] == 2:
                return coordinates
            elif coordinates.ndim == 1 and len(coordinates) % 2 == 0:
                return coordinates.reshape(-1, 2)
            else:
                raise ValueError(
                    "Numpy array must be N×2 or 1D with even length"
                )

        elif isinstance(coordinates, list):
            if len(coordinates) == 0:
                return np.empty((0, 2), dtype=np.float32)

            # Check for list of coordinate pairs
            if all(isinstance(item, (tuple, list)) and len(item) == 2 for item in coordinates):
                return np.array(coordinates, dtype=np.float32)

            # Handle flattened list
            elif len(coordinates) % 2 == 0:
                return np.array(coordinates, dtype=np.float32).reshape(-1, 2)

            else:
                raise ValueError(
                    "List input must contain coordinate pairs or have even length"
                )

        else:
            raise TypeError(
                "Unsupported input type. Use list, tuple, or numpy array"
            )

    def visualize(self, grid=None, start_row=0, start_col=0,
                  height=10, width=10, char_map=(' ', '█')):
        """
        Visualizes a portion of the grid using ASCII characters.

        Args:
            grid (numpy.ndarray, optional): Grid to visualize. Uses internal grid if None.
            start_row (int): Starting row for visualization
            start_col (int): Starting column for visualization
            height (int): Number of rows to display
            width (int): Number of columns to display
            char_map (tuple): Characters to use for (free, occupied) cells
        """
        if grid is None:
            grid = self.grid

        # Calculate safe display boundaries
        end_row = min(start_row + height, self.map_height)
        end_col = min(start_col + width, self.map_width)

        # Display grid section
        for row in range(start_row, end_row):
            line = []
            for col in range(start_col, end_col):
                cell = grid[row, col]
                line.append(char_map[1] if cell == 1 else char_map[0])
            print(''.join(line))


# Example usage
if __name__ == "__main__":
    # Initialize generator with default 100×100 grid
    map_gen = BinaryMapGenerator()

    # Example 1: Create rectangle using list of tuples
    rectangle = [(10, 10), (10, 20), (20, 10), (20, 20)]
    rect_grid = map_gen.generate(rectangle)
    print("Rectangle corners:")
    map_gen.visualize(rect_grid, 5, 5, 15, 15)

    # Example 2: Create triangle using numpy array
    triangle = np.array([[30, 30], [30, 40], [40, 35]])
    tri_grid = map_gen.generate(triangle)
    print("\nTriangle vertices:")
    map_gen.visualize(tri_grid, 25, 25, 15, 15)

    # Example 3: Create circle using flattened list
    circle_points = []
    center_x, center_y = 70, 70
    radius = 5
    for angle in range(0, 360, 18):  # 20 points
        rad = np.deg2rad(angle)
        circle_points.extend([
            center_x + radius * np.cos(rad),
            center_y + radius * np.sin(rad)
        ])

    circle_grid = map_gen.generate(circle_points)
    print("\nCircle points:")
    map_gen.visualize(circle_grid, 60, 60, 15, 15)