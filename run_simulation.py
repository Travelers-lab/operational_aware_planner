from scripts.simulation_test import RobotPlanningTester

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import argparse
import sys
from os.path import join, dirname, abspath

@dataclass
class MapGenerationConfig:
    """Configuration for map generation module."""
    model_checkpoint_path: Optional[str] = None
    model_config_path: Optional[str] = None
    device: str = "cuda:0"


@dataclass
class PerceptionConfig:
    """Configuration for perception module."""
    workspace_bounds: List[List[float]] = field(default_factory=lambda: [[0.30, 0.05], [0.85, 0.6]])
    grid_resolution: int = 100
    contact_threshold: float = 0.005
    min_object_points: int = 10


@dataclass
class OperationLimits:
    """Configuration for operation limits in motion planning."""
    # Add specific operation limit parameters here
    pass


@dataclass
class MotionPlanningConfig:
    """Configuration for motion planning module."""
    operation_limits: OperationLimits = field(default_factory=OperationLimits)


@dataclass
class SimulationTestConfig:
    """Configuration for simulation testing."""
    results_dir: Optional[str] = None
    num_tests: Optional[int] = 100
    robot_urdf_path: str = "single_arm/left_arm.urdf"
    obstacle_count: int = None
    min_obstacle_distance: float = None


@dataclass
class MotionPlanningOverallConfig:
    """Overall configuration for motion planning system."""
    map_generation_config: MapGenerationConfig = field(default_factory=MapGenerationConfig)
    perception_config: PerceptionConfig = field(default_factory=PerceptionConfig)
    motion_planning_config: MotionPlanningConfig = field(default_factory=MotionPlanningConfig)
    simulation_test_config: SimulationTestConfig = field(default_factory=SimulationTestConfig)


def parse_args_with_argparse():
    """
    Example of parsing command line arguments using argparse.
    This is the recommended approach for most applications.
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Motion Planning Configuration Tool",
        epilog="Example: python script.py --config .config/simulation_test_config.yaml --verbose --grid-resolution 150"
    )

    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config/simulation_test_config.yaml',
        help='Path to configuration file (YAML format)'
    )

    # Optional arguments
    parser.add_argument(
        '--grid-resolution',
        type=int,
        default=100,
        help='Grid resolution for perception (default: 100)'
    )

    parser.add_argument(
        '--contact-threshold',
        type=float,
        default=0.005,
        help='Contact threshold value (default: 0.005)'
    )

    parser.add_argument(
        '--workspace',
        type=float,
        nargs=4,  # Expects 4 values: x1, y1, x2, y2
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='Workspace coordinates: x1 y1 x2 y2'
    )

    # Flag arguments (boolean)
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without actual execution'
    )

    # Choice arguments
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'],
        default='cuda:0',
        help='Device to use for computation'
    )

    # Parse arguments
    args = parser.parse_args()

    return args


def load_config(config_path: str = join(dirname(abspath(__file__)), "config/simulation_test-config.yaml")) -> MotionPlanningOverallConfig:
    """
    Load configuration from YAML file and return as config object.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        MotionPlanningOverallConfig: Loaded configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML format
    """
    try:
        with open(join(dirname(abspath(__file__)),config_path), 'r') as file:
            config_data = yaml.safe_load(file)

        # Extract nested configurations
        map_gen_data = config_data.get('motion_planning_config', {}).get('map_generation_config', {})
        perception_data = config_data.get('motion_planning_config', {}).get('perception_config', {})
        motion_planning_data = config_data.get('motion_planning_config', {}).get('motion_planning_config', {})
        simulation_data = config_data.get('motion_planning_config', {}).get('simulation_test_config', {})

        # Create config objects
        map_gen_config = MapGenerationConfig(**map_gen_data)
        perception_config = PerceptionConfig(**perception_data)
        motion_planning_config = MotionPlanningConfig(**motion_planning_data)
        simulation_config = SimulationTestConfig(**simulation_data)

        return MotionPlanningOverallConfig(
            map_generation_config=map_gen_config,
            perception_config=perception_config,
            motion_planning_config=motion_planning_config,
            simulation_test_config=simulation_config
        )

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML format in config file: {e}")



def main():
    """
    Running simulation test among motion planner.
    """

    # Parse command line arguments
    args = parse_args_with_argparse()

    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    # Initialize simulation tester
    planning_test = RobotPlanningTester(config)

    # Running simulation test

    planning_test.run_test_suite()



if __name__ == "__main__":
    main()