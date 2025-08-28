from robot_environment.robot_environment import Robot
import pybullet as p
import pybullet_data as pd
from os.path import join, dirname, abspath
import time

def load_env():
    time_step = 1/200
    client1 = p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setPhysicsEngineParameter(numSolverIterations=50)
    p.setGravity(0, 0, -9.8, physicsClientId=client1)
    p.setTimeStep(time_step)
    robot = Robot(p, join(dirname(abspath(__file__)), "environment_description/single_arm/left_arm.urdf"), client1)
    robot.load_environment([[0.5, 0.3], [0.55, 0.35], [0.65, 0.45]], fix_prob=0.2)
    while True:
        p.stepSimulation()
        time.sleep(time_step)


if __name__ == "__main__":
    load_env()