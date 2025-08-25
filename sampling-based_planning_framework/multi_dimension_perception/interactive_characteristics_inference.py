import numpy as np
from typing import Dict, List, Any, Tuple
from scipy.optimize import least_squares


class MechanicalPropertyInferrer:
    """
    A class for inferring mechanical properties of objects based on interaction data.

    This class processes object interaction data to infer material properties
    using least squares fitting on displacement differences.
    """

    def __init__(self, min_interaction_length: int = 50):
        """
        Initialize the MechanicalPropertyInferrer.

        Args:
            min_interaction_length: Minimum number of interaction pairs required for inference
        """
        self.min_interaction_length = min_interaction_length

    def infer_properties(self, objects_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Infer mechanical properties for all eligible objects in the dictionary.

        Args:
            objects_dict: Dictionary containing object data with the structure:
                {
                    'object_id': {
                        'material_property': [0, 0, 0],
                        'contour_equation': None,
                        'interaction_pairs': n*3*2 array,
                        'contact_status': 'active'
                    }
                }

        Returns:
            Updated objects dictionary with inferred material properties
        """
        # Create a copy to avoid modifying the original dictionary
        updated_objects = objects_dict.copy()

        for obj_id, obj_data in updated_objects.items():
            try:
                # Check if material properties need to be inferred
                if self._should_infer_properties(obj_data):
                    # Preprocess interaction data
                    processed_data = self._preprocess_interaction_data(obj_data['interaction_pairs'])

                    # Perform least squares fitting
                    coefficients = self._fit_mechanical_properties(processed_data)

                    # Update material properties
                    obj_data['material_property'] = coefficients.tolist()

                    print(f"Inferred properties for object {obj_id}: {coefficients}")
                else:
                    print(f"Skipping object {obj_id}: insufficient data or properties already set")

            except Exception as e:
                print(f"Error processing object {obj_id}: {str(e)}")
                continue

        return updated_objects

    def _should_infer_properties(self, obj_data: Dict[str, Any]) -> bool:
        """
        Check if properties should be inferred for this object.

        Args:
            obj_data: Object data dictionary

        Returns:
            bool: True if properties should be inferred, False otherwise
        """
        # Check if material properties are all zeros
        material_property = np.array(obj_data['material_property'])
        if not np.allclose(material_property, 0.0):
            return False

        # Check if interaction pairs exist and have sufficient length
        interaction_pairs = obj_data.get('interaction_pairs', [])
        if not isinstance(interaction_pairs, (list, np.ndarray)) or len(
                interaction_pairs) < self.min_interaction_length:
            return False

        return True

    def _preprocess_interaction_data(self, interaction_pairs: np.ndarray) -> np.ndarray:
        """
        Preprocess interaction data by computing displacement differences and Euclidean distances.

        Args:
            interaction_pairs: Input data of shape n*3*2

        Returns:
            Processed data of shape (n-1)*3 suitable for fitting
        """
        # Convert to numpy array for efficient computation
        data = np.array(interaction_pairs)

        # Verify input shape
        if data.ndim != 3 or data.shape[1] != 3 or data.shape[2] != 2:
            raise ValueError(f"Expected shape n*3*2, got {data.shape}")

        # Compute displacement differences: subtract first element from second element in each pair
        # data shape: (n, 3, 2) where last dimension contains [value1, value2]
        displacement_diffs = data[1:, 0, :] - data[:-1, 0, :]  # Shape: (n-1, 3)
        displacement_diffs = np.concatenate([displacement_diffs[:, np.newaxis, :] , data[1:, 1:, :]], axis=1)

        # Compute Euclidean distances (magnitude of displacement vectors)
        # This gives the distance for each of the 3 dimensions independently
        euclidean_distances = np.sqrt(np.sum(displacement_diffs ** 2, axis=2))  # Shape: (n-1, 3)

        return euclidean_distances

    def _fit_mechanical_properties(self, processed_data: np.ndarray) -> np.ndarray:
        """
        Fit mechanical properties using least squares method.

        The model is: y = a*x1 + b*x2 + c
        where x1 and x2 are input features, y is the target

        Args:
            processed_data: Processed data of shape (n-1)*3

        Returns:
            Array of coefficients [a, b, c]
        """
        n_samples, n_features = processed_data.shape

        if n_samples < 2:
            raise ValueError("Insufficient data samples for fitting")

        if n_features != 3:
            raise ValueError(f"Expected 3 features, got {n_features}")

        # Prepare data for linear regression: y = a*x1 + b*x2 + c
        # Use first two features as inputs (x1, x2), third feature as output (y)
        X = processed_data[:, :2]  # Input features: first two columns
        y = processed_data[:, 2]  # Output: third column

        # Add column of ones for the intercept term
        A = np.column_stack([X, np.ones(len(X))])

        # Solve using least squares: min ||A * coefficients - y||^2
        coefficients, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

        return coefficients

    def infer_properties_single(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer mechanical properties for a single object.

        Args:
            obj_data: Single object data dictionary

        Returns:
            Updated object data with inferred material properties
        """
        if self._should_infer_properties(obj_data):
            processed_data = self._preprocess_interaction_data(obj_data['interaction_pairs'])
            coefficients = self._fit_mechanical_properties(processed_data)

            # Create a copy of the object data
            updated_obj = obj_data.copy()
            updated_obj['material_property'] = coefficients.tolist()

            return updated_obj
        else:
            return obj_data


# Example usage
if __name__ == "__main__":
    # Create sample objects dictionary
    sample_objects = {
        'object_1': {
            'material_property': [0, 0, 0],
            'contour_equation': None,
            'interaction_pairs': np.random.randn(60, 3, 2),  # 60 samples, 3 features, 2 values each
            'contact_status': 'active'
        },
        'object_2': {
            'material_property': [1.0, 2.0, 3.0],  # Already has properties
            'contour_equation': None,
            'interaction_pairs': np.random.randn(40, 3, 2),  # Insufficient length
            'contact_status': 'active'
        },
        'object_3': {
            'material_property': [0, 0, 0],
            'contour_equation': None,
            'interaction_pairs': np.random.randn(55, 3, 2),  # 55 samples
            'contact_status': 'inactive'
        }
    }

    # Initialize the inferrer
    inferrer = MechanicalPropertyInferrer(min_interaction_length=50)

    # Infer properties for all objects
    updated_objects = inferrer.infer_properties(sample_objects)

    # Display results
    for obj_id, obj_data in updated_objects.items():
        print(f"{obj_id}: material_property = {obj_data['material_property']}")