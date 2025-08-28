from typing import Dict, List, Any, Tuple, Optional
import numpy as np


class MotionIntentRecognizer:
    """
    A class for recognizing motion intent based on object contact status and properties.

    This class determines the appropriate task state and returns target points
    for the planner based on object interactions and material properties.
    """

    def __init__(self):
        """
        Initialize the Motion Intent Recognizer.
        """
        self.recognized_shapes = ['square', 'circle', 'triangle']

    def get_target_point(self,
                         objects_dict: Dict[str, Dict[str, Any]],
                         motion_target: List[float]) -> List[float]:
        """
        Determine the target point based on object contact status and properties.

        Args:
            objects_dict: Dictionary containing object data with the structure:
                {
                    'object_id': {
                        'material_property': [float, float, float],
                        'contour_equation': dict or None,
                        'interaction_pairs': list,
                        'contact_status': str ('active', 'inactive', 'contact')
                    }
                }
            motion_target: Original motion target point as 1x2 numpy array

        Returns:
            Target point for the planner as 1x2 numpy array
        """
        # Validate input
        print(motion_target, type(motion_target))
        if not isinstance(motion_target, list) or len(motion_target) != 2:
            raise ValueError("motion_target must be a 1x2 list")

        if not objects_dict:
            print("No objects provided, returning original motion target")
            return motion_target

        # Check each object for contact and material properties
        for obj_id, obj_data in objects_dict.items():
            try:
                # Check if object is in contact and has default material properties
                if self._should_process_object(obj_data):
                    print(f"Processing object {obj_id} with contact status: {obj_data['contact_status']}")

                    # Extract center point from contour equation
                    center_point = self._extract_center_from_contour(obj_data['contour_equation'])

                    if center_point is not None:
                        print(f"Found center point {center_point} for object {obj_id}")
                        return center_point
                    else:
                        print(f"Could not extract center from contour equation for object {obj_id}")

            except Exception as e:
                print(f"Error processing object {obj_id}: {str(e)}")
                continue

        # If no contact object with default properties found, return original target
        print("No suitable contact object found, returning original motion target")
        return motion_target

    def _should_process_object(self, obj_data: Dict[str, Any]) -> bool:
        """
        Check if an object should be processed for target point extraction.

        Args:
            obj_data: Object data dictionary

        Returns:
            bool: True if object should be processed, False otherwise
        """
        # Check if object is in contact
        contact_status = obj_data.get('contact_status', '').lower()
        if contact_status not in ['active', 'contact']:
            return False

        # Check if material properties are default (all zeros)
        material_property = obj_data.get('material_property', [1, 1, 1])  # Default to non-zero if missing
        if not isinstance(material_property, (list, np.ndarray)) or len(material_property) != 3:
            return False

        # Check if all material properties are approximately zero
        material_array = np.array(material_property, dtype=float)
        if not np.allclose(material_array, 0.0, atol=1e-6):
            return False

        # Check if contour equation exists and is valid
        contour_equation = obj_data.get('contour_equation')
        if contour_equation is None or not isinstance(contour_equation, dict):
            return False

        return True

    def _extract_center_from_contour(self, contour_equation: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extract center point from contour equation based on shape type.

        Args:
            contour_equation: Dictionary containing shape parameters
                Example formats:
                Square: {'shape': 'square', 'center': [x, y], 'side_length': float}
                Circle: {'shape': 'circle', 'center': [x, y], 'radius': float}
                Triangle: {'shape': 'triangle', 'vertices': [[x1,y1], [x2,y2], [x3,y3]]}

        Returns:
            Center point as 1x2 numpy array, or None if extraction fails
        """
        if not isinstance(contour_equation, dict):
            return None

        shape_type = contour_equation.get('shape', '').lower()

        try:
            if shape_type == 'square':
                center = contour_equation.get('center')
                if center and len(center) == 2:
                    return center

            elif shape_type == 'circle':
                center = contour_equation.get('center')
                if center and len(center) == 2:
                    return center

            elif shape_type == 'triangle':
                vertices = contour_equation.get('vertices')
                if vertices and len(vertices) == 3:
                    # Calculate centroid of triangle
                    vertices_array = np.array(vertices, dtype=float)
                    if vertices_array.shape == (3, 2):
                        centroid = np.mean(vertices_array, axis=0)
                        return centroid

            # Handle other shape representations
            elif 'center' in contour_equation:
                center = contour_equation['center']
                if isinstance(center, (list, np.ndarray)) and len(center) == 2:
                    return center

            print(f"Unsupported or incomplete contour equation for shape: {shape_type}")
            return None

        except (ValueError, TypeError) as e:
            print(f"Error extracting center from contour equation: {str(e)}")
            return None

    def get_task_state(self,
                       objects_dict: Dict[str, Dict[str, Any]],
                       motion_target: np.ndarray) -> Dict[str, Any]:
        """
        Get complete task state information including target point and reasoning.

        Args:
            objects_dict: Dictionary containing object data
            motion_target: Original motion target point as 1x2 numpy array

        Returns:
            Dictionary containing:
                - target_point: Final target point
                - task_state: Description of current task state
                - processed_object: ID of the object that was processed (if any)
                - reason: Explanation for the decision
        """
        original_target = motion_target.copy()

        # Find contact objects with default properties
        contact_objects = []
        for obj_id, obj_data in objects_dict.items():
            if self._should_process_object(obj_data):
                contact_objects.append((obj_id, obj_data))

        if not contact_objects:
            return {
                'target_point': original_target,
                'task_state': 'free_motion',
                'processed_object': None,
                'reason': 'No objects in contact with default material properties'
            }

        # Process the first suitable object
        obj_id, obj_data = contact_objects[0]
        center_point = self._extract_center_from_contour(obj_data['contour_equation'])

        if center_point is not None:
            return {
                'target_point': center_point,
                'task_state': 'object_interaction',
                'processed_object': obj_id,
                'reason': f'Object {obj_id} in contact with default properties, moving to center'
            }
        else:
            return {
                'target_point': original_target,
                'task_state': 'free_motion',
                'processed_object': obj_id,
                'reason': f'Object {obj_id} in contact but could not extract center point'
            }


# Example usage
if __name__ == "__main__":
    # Initialize the motion intent recognizer
    recognizer = MotionIntentRecognizer()

    # Create sample objects dictionary
    sample_objects = {
        'object_1': {
            'material_property': [0, 0, 0],
            'contour_equation': {
                'shape': 'square',
                'center': [1.5, 2.0],
                'side_length': 1.0
            },
            'interaction_pairs': [],
            'contact_status': 'contact'
        },
        'object_2': {
            'material_property': [1.0, 2.0, 3.0],
            'contour_equation': {
                'shape': 'circle',
                'center': [3.0, 4.0],
                'radius': 0.5
            },
            'interaction_pairs': [],
            'contact_status': 'active'
        },
        'object_3': {
            'material_property': [0, 0, 0],
            'contour_equation': {
                'shape': 'triangle',
                'vertices': [[0, 0], [1, 0], [0.5, 1]]
            },
            'interaction_pairs': [],
            'contact_status': 'inactive'
        }
    }

    # Original motion target
    motion_target = np.array([5.0, 5.0])

    # Get target point
    target_point = recognizer.get_target_point(sample_objects, motion_target)
    print(f"Final target point: {target_point}")

    # Get complete task state information
    task_state = recognizer.get_task_state(sample_objects, motion_target)
    print(f"Task state: {task_state}")