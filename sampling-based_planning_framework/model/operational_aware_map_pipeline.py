import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
import logging
from map_generation import BinaryMapGenerator
from model_prediction import MapCompletionPredictor
from operational_cost_map_generation import OperationCostMapGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MapGenerationPipeline:
    """
    A pipeline class for generating operational cost maps from point cloud data.

    This class integrates three components:
    1. BinaryMapGenerator: Generates incomplete binary maps from point clouds
    2. MapCompletionPredictor: Completes binary maps using a neural network
    3. OperationCostMapGenerator: Generates operational cost maps from completed binary maps
    """

    def __init__(self,
                 model_checkpoint_path: str,
                 operation_limits: List[float] = [400, 400, 20],
                 device: Optional[str] = None,
                 model_config_path: Optional[str] = None):
        """
        Initialize the map generation pipeline.

        Args:
            model_checkpoint_path: Path to the trained model checkpoint
            operation_limits: Operation limits for cost calculation [limit1, limit2, limit3]
            device: Device for model inference ('cuda', 'cpu', or None for auto-detection)
            model_config_path: Optional path to model configuration file
        """
        # Initialize components
        self.binary_generator = BinaryMapGenerator()
        self.map_predictor = MapCompletionPredictor(
            checkpoint_path=model_checkpoint_path,
            device=device,
            config_path=model_config_path
        )
        self.cost_generator = OperationCostMapGenerator(operation_limits=operation_limits)

        logger.info("MapGenerationPipeline initialized successfully")

    def generate_operational_cost_map(self,
                                      point_cloud_coordinates: List[List[float]],
                                      objects_dict: Dict[str, Dict[str, Any]],
                                      return_intermediate: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate operational cost map from point cloud coordinates and object properties.

        Args:
            point_cloud_coordinates: List of point cloud coordinates [[x1, y1], [x2, y2], ...]
            objects_dict: Dictionary containing object properties
            return_intermediate: If True, return intermediate maps along with final cost map

        Returns:
            Operational cost map or dictionary containing all intermediate maps
        """
        try:
            # Step 1: Generate incomplete binary map from point cloud
            # logger.info("Generating incomplete binary map from point cloud...")
            incomplete_map = self.binary_generator.generate(coordinates=point_cloud_coordinates)

            if incomplete_map is None:
                raise ValueError("Failed to generate incomplete binary map")

            # logger.info(f"Incomplete map generated with shape: {incomplete_map.shape}")

            # Step 2: Complete the binary map using neural network prediction
            # logger.info("Completing binary map using neural network...")
            completed_map = self.map_predictor.predict(incomplete_map)

            if completed_map is None:
                raise ValueError("Failed to complete binary map")

            # logger.info(f"Completed map generated with shape: {completed_map.shape}")

            # Step 3: Generate operational cost map from completed binary map
            # logger.info("Generating operational cost map...")
            cost_map = self.cost_generator.generate_cost_map(
                binary_map=completed_map,
                objects_dict=objects_dict
            )

            if cost_map is None:
                raise ValueError("Failed to generate operational cost map")

            # logger.info(f"Operational cost map generated with shape: {cost_map.shape}")
            # logger.info(f"Cost map range: [{cost_map.min():.3f}, {cost_map.max():.3f}]")

            if return_intermediate:
                return {
                    'incomplete_map': incomplete_map,
                    'completed_map': completed_map,
                    'operational_cost_map': cost_map,
                    'point_cloud_coordinates': point_cloud_coordinates
                }
            else:
                return cost_map, completed_map

        except Exception as e:
            logger.error(f"Error in map generation pipeline: {str(e)}")
            raise RuntimeError(f"Map generation failed: {str(e)}")

    def batch_generate_cost_maps(self,
                                 batch_point_clouds: List[List[List[float]]],
                                 batch_objects_dicts: List[Dict[str, Dict[str, Any]]],
                                 batch_size: int = 4) -> List[np.ndarray]:
        """
        Generate operational cost maps for multiple point cloud sets in batch.

        Args:
            batch_point_clouds: List of point cloud coordinate lists
            batch_objects_dicts: List of objects dictionaries
            batch_size: Number of maps to process in each batch

        Returns:
            List of operational cost maps
        """
        if len(batch_point_clouds) != len(batch_objects_dicts):
            raise ValueError("Number of point cloud sets must match number of objects dictionaries")

        all_cost_maps = []

        for i in range(0, len(batch_point_clouds), batch_size):
            batch_end = min(i + batch_size, len(batch_point_clouds))
            logger.info(f"Processing batch {i // batch_size + 1}: items {i} to {batch_end - 1}")

            for j in range(i, batch_end):
                try:
                    cost_map = self.generate_operational_cost_map(
                        point_cloud_coordinates=batch_point_clouds[j],
                        objects_dict=batch_objects_dicts[j],
                        return_intermediate=False
                    )
                    all_cost_maps.append(cost_map)

                except Exception as e:
                    logger.error(f"Failed to process item {j}: {str(e)}")
                    # Append empty map or continue based on requirements
                    all_cost_maps.append(np.zeros((100, 100), dtype=np.float32))

        return all_cost_maps

    def visualize_pipeline_results(self,
                                   point_cloud_coordinates: List[List[float]],
                                   objects_dict: Dict[str, Dict[str, Any]],
                                   save_path: Optional[str] = None):
        """
        Visualize all intermediate results of the pipeline.

        Args:
            point_cloud_coordinates: List of point cloud coordinates
            objects_dict: Dictionary containing object properties
            save_path: Optional path to save visualization
        """
        try:
            # Generate all intermediate maps
            results = self.generate_operational_cost_map(
                point_cloud_coordinates=point_cloud_coordinates,
                objects_dict=objects_dict,
                return_intermediate=True
            )

            # Import here to avoid dependency if not used
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Plot point cloud (scatter plot)
            if results['point_cloud_coordinates']:
                coords = np.array(results['point_cloud_coordinates'])
                axes[0, 0].scatter(coords[:, 0], coords[:, 1], s=1, c='blue', alpha=0.6)
                axes[0, 0].set_title('Input Point Cloud')
                axes[0, 0].set_xlim(0, 100)
                axes[0, 0].set_ylim(0, 100)
                axes[0, 0].set_aspect('equal')

            # Plot incomplete map
            im1 = axes[0, 1].imshow(results['incomplete_map'], cmap='binary', vmin=0, vmax=1)
            axes[0, 1].set_title('Incomplete Binary Map')
            plt.colorbar(im1, ax=axes[0, 1])

            # Plot completed map
            im2 = axes[1, 0].imshow(results['completed_map'], cmap='binary', vmin=0, vmax=1)
            axes[1, 0].set_title('Completed Binary Map')
            plt.colorbar(im2, ax=axes[1, 0])

            # Plot operational cost map
            im3 = axes[1, 1].imshow(results['operational_cost_map'], cmap='hot', vmin=0, vmax=1)
            axes[1, 1].set_title('Operational Cost Map')
            plt.colorbar(im3, ax=axes[1, 1])

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline components.

        Returns:
            Dictionary containing pipeline component information
        """
        info = {
            'pipeline_type': 'MapGenerationPipeline',
            'components': {
                'binary_generator': 'BinaryMapGenerator',
                'map_predictor': 'MapCompletionPredictor',
                'cost_generator': 'OperationCostMapGenerator'
            },
            'map_size': (100, 100),
            'output_type': 'operational_cost_map'
        }

        # Add model info if available
        try:
            model_info = self.map_predictor.get_model_info()
            info['model_info'] = model_info
        except:
            info['model_info'] = 'Not available'

        return info


# Example usage
if __name__ == "__main__":
    # Example usage of the pipeline
    try:
        # Initialize pipeline
        pipeline = MapGenerationPipeline(
            model_checkpoint_path="path/to/model_checkpoint.pth",
            operation_limits=[400, 400, 20],
            device="cuda"  # Auto-detect if None
        )

        # Get pipeline information
        pipeline_info = pipeline.get_pipeline_info()
        print("Pipeline Info:", pipeline_info)

        # Create sample point cloud data (random for demonstration)
        sample_point_cloud = []
        for _ in range(200):
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 100)
            sample_point_cloud.append([x, y])

        # Create sample objects dictionary
        sample_objects = {
            'object_1': {
                'material_property': [200, 150, 8],
                'contour_equation': {
                    'shape': 'square',
                    'params': {'center': [35, 35], 'side_length': 20}
                },
                'interaction_pairs': [],
                'contact_status': 'active'
            }
        }

        # Generate operational cost map
        cost_map = pipeline.generate_operational_cost_map(
            point_cloud_coordinates=sample_point_cloud,
            objects_dict=sample_objects
        )

        print(f"Generated cost map with shape: {cost_map.shape}")
        print(
            f"Cost map statistics - Min: {cost_map.min():.3f}, Max: {cost_map.max():.3f}, Mean: {cost_map.mean():.3f}")

        # Visualize results
        pipeline.visualize_pipeline_results(
            point_cloud_coordinates=sample_point_cloud,
            objects_dict=sample_objects,
            save_path="pipeline_results.png"
        )

    except Exception as e:
        print(f"Error in pipeline execution: {e}")