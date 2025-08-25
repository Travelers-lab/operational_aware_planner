import numpy as np
from typing import List, Tuple, Optional
from scipy import interpolate
import warnings


class SamplingPointOptimizer:
    """
    A class for optimizing sampled points from motion planning by performing smooth interpolation.

    This class takes sampled points from a motion planner and performs interpolation
    to create a smooth path with consistent Euclidean distance between consecutive points.

    Attributes:
        interpolation_distance (float): The desired Euclidean distance between interpolated points.
        interpolation_method (str): The method used for interpolation ('linear', 'cubic', etc.).
        min_points_for_interpolation (int): Minimum number of points required for interpolation.
    """

    def __init__(self,
                 interpolation_distance: float = 0.0002,
                 interpolation_method: str = 'linear',
                 min_points_for_interpolation: int = 2):
        """
        Initialize the SamplingPointOptimizer.

        Args:
            interpolation_distance: Desired Euclidean distance between interpolated points (default: 0.0002)
            interpolation_method: Interpolation method ('linear', 'cubic', 'quadratic') (default: 'linear')
            min_points_for_interpolation: Minimum number of points required for interpolation (default: 2)

        Raises:
            ValueError: If interpolation_distance is not positive or min_points_for_interpolation is less than 2
        """
        if interpolation_distance <= 0:
            raise ValueError("interpolation_distance must be a positive number")
        if min_points_for_interpolation < 2:
            raise ValueError("min_points_for_interpolation must be at least 2")

        self.interpolation_distance = interpolation_distance
        self.interpolation_method = interpolation_method
        self.min_points_for_interpolation = min_points_for_interpolation

    def _validate_input_points(self, sampled_points: List[np.ndarray]) -> None:
        """
        Validate the input sampled points.

        Args:
            sampled_points: List of numpy arrays representing sampled points

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(sampled_points, list):
            raise ValueError("sampled_points must be a list")

        if len(sampled_points) < self.min_points_for_interpolation:
            raise ValueError(f"At least {self.min_points_for_interpolation} points are required for interpolation")

        # Check all points have the same dimension
        dimensions = [point.shape[0] for point in sampled_points]
        if len(set(dimensions)) > 1:
            raise ValueError("All sampled points must have the same dimension")

        # Check if points are distinct
        for i in range(len(sampled_points) - 1):
            if np.array_equal(sampled_points[i], sampled_points[i + 1]):
                warnings.warn(f"Consecutive points at indices {i} and {i + 1} are identical")

    def _calculate_cumulative_distance(self, points: List[np.ndarray]) -> np.ndarray:
        """
        Calculate cumulative distance along the path.

        Args:
            points: List of points

        Returns:
            Cumulative distance array
        """
        cumulative_distances = [0.0]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i - 1])
            cumulative_distances.append(cumulative_distances[-1] + dist)

        return np.array(cumulative_distances)

    def _interpolate_segment(self,
                             start_point: np.ndarray,
                             end_point: np.ndarray,
                             segment_length: float) -> List[np.ndarray]:
        """
        Interpolate between two points with specified spacing.

        Args:
            start_point: Starting point of the segment
            end_point: Ending point of the segment
            segment_length: Length of the segment

        Returns:
            List of interpolated points including start and end points
        """
        if segment_length < self.interpolation_distance:
            return [start_point, end_point]

        # Calculate number of interpolation points
        num_points = max(2, int(np.ceil(segment_length / self.interpolation_distance)) + 1)

        # Linear interpolation between start and end points
        interpolated_points = []
        for i in range(num_points):
            alpha = i / (num_points - 1)
            point = start_point + alpha * (end_point - start_point)
            interpolated_points.append(point)

        return interpolated_points

    def _spline_interpolation(self, sampled_points: List[np.ndarray]) -> List[np.ndarray]:
        """
        Perform spline interpolation on sampled points.

        Args:
            sampled_points: List of sampled points

        Returns:
            List of interpolated points
        """
        # Convert to numpy array for easier manipulation
        points_array = np.array(sampled_points)
        n_points, n_dimensions = points_array.shape

        # Calculate cumulative distance along the path
        cumulative_dist = self._calculate_cumulative_distance(sampled_points)
        total_length = cumulative_dist[-1]

        # Create interpolation functions for each dimension
        interp_functions = []
        for dim in range(n_dimensions):
            if self.interpolation_method == 'linear':
                interp_func = interpolate.interp1d(cumulative_dist, points_array[:, dim],
                                                   kind='linear', fill_value='extrapolate')
            else:
                interp_func = interpolate.interp1d(cumulative_dist, points_array[:, dim],
                                                   kind=self.interpolation_method,
                                                   fill_value='extrapolate')
            interp_functions.append(interp_func)

        # Generate new points with consistent spacing
        num_new_points = max(2, int(np.ceil(total_length / self.interpolation_distance)) + 1)
        new_distances = np.linspace(0, total_length, num_new_points)

        # Interpolate each dimension
        interpolated_points = []
        for dist in new_distances:
            point = np.array([func(dist) for func in interp_functions])
            interpolated_points.append(point)

        return interpolated_points

    def optimize_points(self, sampled_points: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optimize sampled points by performing smooth interpolation.

        Args:
            sampled_points: List of numpy arrays representing sampled points from motion planning

        Returns:
            List of interpolated points with consistent Euclidean distance

        Raises:
            ValueError: If input points are invalid
        """
        # Validate input points
        self._validate_input_points(sampled_points)

        # If only two points, use simple linear interpolation
        if len(sampled_points) == 2:
            start_point = sampled_points[0]
            end_point = sampled_points[1]
            segment_length = np.linalg.norm(end_point - start_point)
            return self._interpolate_segment(start_point, end_point, segment_length)

        # For more than two points, use spline interpolation
        try:
            return self._spline_interpolation(sampled_points)
        except Exception as e:
            warnings.warn(f"Spline interpolation failed: {e}. Falling back to piecewise linear interpolation.")
            return self._piecewise_linear_interpolation(sampled_points)

    def _piecewise_linear_interpolation(self, sampled_points: List[np.ndarray]) -> List[np.ndarray]:
        """
        Fallback method: piecewise linear interpolation between consecutive points.

        Args:
            sampled_points: List of sampled points

        Returns:
            List of interpolated points
        """
        interpolated_path = []

        for i in range(len(sampled_points) - 1):
            start_point = sampled_points[i]
            end_point = sampled_points[i + 1]
            segment_length = np.linalg.norm(end_point - start_point)

            # Interpolate current segment
            segment_points = self._interpolate_segment(start_point, end_point, segment_length)

            # Add points (excluding the last point to avoid duplication)
            if i == 0:
                interpolated_path.extend(segment_points)
            else:
                interpolated_path.extend(segment_points[1:])

        return interpolated_path

    def calculate_path_quality(self, optimized_points: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate quality metrics for the optimized path.

        Args:
            optimized_points: List of optimized points

        Returns:
            Dictionary containing quality metrics
        """
        if len(optimized_points) < 2:
            return {"average_distance": 0.0, "max_distance": 0.0, "min_distance": 0.0}

        distances = []
        for i in range(len(optimized_points) - 1):
            dist = np.linalg.norm(optimized_points[i + 1] - optimized_points[i])
            distances.append(dist)

        distances = np.array(distances)

        return {
            "average_distance": float(np.mean(distances)),
            "max_distance": float(np.max(distances)),
            "min_distance": float(np.min(distances)),
            "std_distance": float(np.std(distances)),
            "total_points": len(optimized_points),
            "total_length": float(np.sum(distances))
        }


# Example usage
if __name__ == "__main__":
    # Create sample points
    sampled_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.2, 0.1]),
        np.array([0.3, 0.4, 0.2]),
        np.array([0.6, 0.5, 0.3])
    ]

    # Initialize optimizer
    optimizer = SamplingPointOptimizer(interpolation_distance=0.0002, interpolation_method='cubic')

    # Optimize points
    optimized_points = optimizer.optimize_points(sampled_points)

    # Calculate quality metrics
    quality_metrics = optimizer.calculate_path_quality(optimized_points)

    print(f"Original points: {len(sampled_points)}")
    print(f"Optimized points: {len(optimized_points)}")
    print(f"Quality metrics: {quality_metrics}")