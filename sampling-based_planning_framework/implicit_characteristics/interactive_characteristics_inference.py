import numpy as np
from typing import Tuple, List, Optional


def least_squares_estimation(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate parameters of a linear equation (y = a*x1 + b*x2 + c) using least squares method.

    This function solves the linear system using the normal equation: (X^T * X) * params = X^T * y

    Args:
        data (np.ndarray): Input data array of shape (n, 3) where each row is [x1, x2, y]

    Returns:
        Tuple[float, float, float]: Estimated parameters (a, b, c) for the equation y = a*x1 + b*x2 + c

    Raises:
        ValueError: If input data has incorrect shape or is singular
    """
    # Validate input data
    if data.shape[1] != 3:
        raise ValueError(f"Input data must have shape (n, 3), got {data.shape}")

    if len(data) < 3:
        raise ValueError(f"At least 3 data points are required, got {len(data)}")

    # Extract features (x1, x2) and target (y)
    X = data[:, :2]  # x1 and x2 columns
    y = data[:, 2]  # y column

    # Add column of ones for the intercept term (c)
    X_augmented = np.column_stack([X, np.ones(len(X))])

    try:
        # Solve using normal equation: params = (X^T * X)^(-1) * X^T * y
        XTX = np.dot(X_augmented.T, X_augmented)
        XTy = np.dot(X_augmented.T, y)
        params = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse if matrix is singular
        try:
            params = np.linalg.lstsq(X_augmented, y, rcond=None)[0]
        except np.linalg.LinAlgError as e:
            raise ValueError("Matrix is singular and cannot be solved") from e

    # Extract parameters
    a, b, c = params[0], params[1], params[2]

    return a, b, c


def calculate_r_squared(data: np.ndarray, a: float, b: float, c: float) -> float:
    """
    Calculate R-squared value to measure goodness of fit.

    Args:
        data (np.ndarray): Input data array of shape (n, 3)
        a (float): Parameter a from the equation
        b (float): Parameter b from the equation
        c (float): Parameter c from the equation

    Returns:
        float: R-squared value between 0 and 1
    """
    X = data[:, :2]
    y_actual = data[:, 2]

    # Calculate predicted values
    y_predicted = a * X[:, 0] + b * X[:, 1] + c

    # Calculate R-squared
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return r_squared


def robust_least_squares(data: np.ndarray, max_iterations: int = 100, threshold: float = 2.0) -> Tuple[
    float, float, float]:
    """
    Perform robust least squares estimation using iterative reweighting.

    This method is less sensitive to outliers than standard least squares.

    Args:
        data (np.ndarray): Input data array of shape (n, 3)
        max_iterations (int): Maximum number of iterations
        threshold (float): Threshold for outlier detection

    Returns:
        Tuple[float, float, float]: Robust estimated parameters (a, b, c)
    """
    # Initial estimation using standard least squares
    a, b, c = least_squares_estimation(data)

    for iteration in range(max_iterations):
        # Calculate residuals
        X = data[:, :2]
        y_actual = data[:, 2]
        residuals = y_actual - (a * X[:, 0] + b * X[:, 1] + c)

        # Calculate weights using Huber loss function
        weights = np.ones_like(residuals)
        outlier_mask = np.abs(residuals) > threshold * np.std(residuals)
        weights[outlier_mask] = threshold * np.std(residuals) / np.abs(residuals[outlier_mask])

        # Perform weighted least squares
        X_augmented = np.column_stack([X, np.ones(len(X))])
        try:
            XTWX = np.dot(X_augmented.T * weights, X_augmented)
            XTWy = np.dot(X_augmented.T * weights, y_actual)
            new_params = np.linalg.solve(XTWX, XTWy)
        except np.linalg.LinAlgError:
            break

        # Check for convergence
        if np.allclose([a, b, c], new_params, rtol=1e-6):
            break

        a, b, c = new_params

    return a, b, c


# Example usage and test functions
def test_least_squares():
    """
    Test the least squares estimation with sample data.
    """
    # Generate sample data with known parameters: y = 2*x1 + 3*x2 + 5
    np.random.seed(42)  # For reproducible results
    n_points = 100

    x1 = np.random.randn(n_points)
    x2 = np.random.randn(n_points)
    noise = 0.1 * np.random.randn(n_points)
    y = 2 * x1 + 3 * x2 + 5 + noise

    data = np.column_stack([x1, x2, y])

    # Estimate parameters
    try:
        a, b, c = least_squares_estimation(data)
        r_squared = calculate_r_squared(data, a, b, c)

        print("Least Squares Estimation Results:")
        print(f"Estimated equation: y = {a:.4f}*x1 + {b:.4f}*x2 + {c:.4f}")
        print(f"True equation: y = 2.0000*x1 + 3.0000*x2 + 5.0000")
        print(f"R-squared: {r_squared:.6f}")
        print(f"Mean absolute error: {np.mean(np.abs(y - (a * x1 + b * x2 + c))):.6f}")

        # Test with outliers
        data_with_outliers = data.copy()
        data_with_outliers[::10, 2] += 10  # Add large outliers every 10th point

        a_robust, b_robust, c_robust = robust_least_squares(data_with_outliers)
        r_squared_robust = calculate_r_squared(data, a_robust, b_robust, c_robust)

        print("\nRobust Estimation Results (with outliers):")
        print(f"Estimated equation: y = {a_robust:.4f}*x1 + {b_robust:.4f}*x2 + {c_robust:.4f}")
        print(f"R-squared: {r_squared_robust:.6f}")

    except ValueError as e:
        print(f"Error: {e}")


def validate_estimation(data: np.ndarray, a: float, b: float, c: float) -> dict:
    """
    Validate the estimation results with various metrics.

    Args:
        data (np.ndarray): Input data
        a, b, c (float): Estimated parameters

    Returns:
        dict: Dictionary containing validation metrics
    """
    X = data[:, :2]
    y_actual = data[:, 2]
    y_predicted = a * X[:, 0] + b * X[:, 1] + c

    residuals = y_actual - y_predicted

    metrics = {
        'mse': np.mean(residuals ** 2),
        'mae': np.mean(np.abs(residuals)),
        'rmse': np.sqrt(np.mean(residuals ** 2)),
        'r_squared': calculate_r_squared(data, a, b, c),
        'max_error': np.max(np.abs(residuals)),
        'residual_std': np.std(residuals)
    }

    return metrics


if __name__ == "__main__":
    # Run test
    test_least_squares()

    # Example with custom data
    print("\n" + "=" * 50)
    print("Custom Data Example:")

    # Create some sample data
    custom_data = np.array([
        [1, 2, 10],  # y = a*1 + b*2 + c = 10
        [2, 3, 16],  # y = a*2 + b*3 + c = 16
        [3, 1, 12],  # y = a*3 + b*1 + c = 12
        [4, 4, 22],  # y = a*4 + b*4 + c = 22
        [0, 5, 15]  # y = a*0 + b*5 + c = 15
    ])

    try:
        a, b, c = least_squares_estimation(custom_data)
        metrics = validate_estimation(custom_data, a, b, c)

        print(f"Estimated equation: y = {a:.4f}*x1 + {b:.4f}*x2 + {c:.4f}")
        print("Validation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")

    except ValueError as e:
        print(f"Error: {e}")