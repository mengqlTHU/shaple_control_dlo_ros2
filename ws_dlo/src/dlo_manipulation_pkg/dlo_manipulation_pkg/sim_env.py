#!/usr/bin/env python

# main node of the simulation
# bridge between the Unity Simulator and the controller scripts, based on 'mlagents' and 'gym'

import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import rclpy
from rclpy.node import Node
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import gym
from gym_unity.envs import UnityToGymWrapper
from .utils.state_index import Index
from geometry_msgs.msg import Vector3

from .controller_ours import Controller



class Environment(Node):
    def __init__(self):
        super().__init__('sim_env')

        self.declare_parameter('project_dir', rclpy.Parameter.Type.STRING)
        self.declare_parameter('env/dimension', rclpy.Parameter.Type.STRING)
        self.declare_parameter('DLO/num_FPs', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('env/sim_or_real', rclpy.Parameter.Type.STRING) 
        self.declare_parameter('controller/enable_end_rotation', rclpy.Parameter.Type.BOOL)
        self.declare_parameter('controller/enable_left_arm', rclpy.Parameter.Type.BOOL)
        self.declare_parameter('controller/enable_right_arm', rclpy.Parameter.Type.BOOL)
        self.declare_parameter('learning/is_test', rclpy.Parameter.Type.BOOL)
        self.declare_parameter('controller/object_fps_idx', rclpy.Parameter.Type.INTEGER_ARRAY)
        self.declare_parameter('controller/offline_model', rclpy.Parameter.Type.STRING)
        self.declare_parameter('controller/control_law', rclpy.Parameter.Type.STRING)
        self.declare_parameter('ros_rate/env_rate', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('ros_rate/online_update_rate', rclpy.Parameter.Type.INTEGER) 
        self.declare_parameter('controller/online_learning/learning_rate', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('controller/online_learning/weight_ratio', rclpy.Parameter.Type.INTEGER) 

        self.project_dir = self.get_parameter("project_dir").get_parameter_value().string_value
        env_dim = self.get_parameter("env/dimension").get_parameter_value().string_value
        self.num_fps = self.get_parameter("DLO/num_FPs").get_parameter_value().integer_value

        self.get_logger().info(f"fps_idx:{self.get_parameter('controller/object_fps_idx').value}")

        controller_param_dict ={
            "project_dir":self.project_dir,
            "env":self.get_parameter('env/sim_or_real').get_parameter_value().string_value,
            "env_dim" : env_dim,
            "num_fps":self.num_fps,
            "bEnableEndRotation":self.get_parameter('controller/enable_end_rotation').get_parameter_value().bool_value,
            "b_left_arm":self.get_parameter('controller/enable_left_arm').get_parameter_value().bool_value,
            "b_right_arm" : self.get_parameter('controller/enable_right_arm').get_parameter_value().bool_value,
            "targetFPsIdx" : self.get_parameter('controller/object_fps_idx').value,
            "project_dir" : self.project_dir,
            "offline_model_name" : self.get_parameter('controller/offline_model').get_parameter_value().string_value,
            "control_law" : self.get_parameter('controller/control_law').get_parameter_value().string_value,
            "controlRate" : self.get_parameter('ros_rate/env_rate').get_parameter_value().integer_value,
            "online_learning_rate" : self.get_parameter("controller/online_learning/learning_rate").get_parameter_value().double_value,
            "weight_ratio": self.get_parameter("controller/online_learning/weight_ratio").get_parameter_value().integer_value,
            "online_update_rate" : self.get_parameter("ros_rate/online_update_rate").get_parameter_value().integer_value,
            "learning_istest": self.get_parameter("learning/is_test").get_parameter_value().bool_value
        }

        engine_config_channel = EngineConfigurationChannel()
        env_params_channel = EnvironmentParametersChannel()

        # use the built Unity environment
        env_file = self.project_dir + "env_dlo/env_" + env_dim
        unity_env = UnityEnvironment(file_name=env_file, seed=1, side_channels=[engine_config_channel, env_params_channel])
        engine_config_channel.set_configuration_parameters(width=640, height=360, time_scale=2.0)  # speed x2
        
        self.env = UnityToGymWrapper(unity_env)
        self.get_logger().info(f"controller_param_dict:{controller_param_dict}")
        self.controller = Controller(params_dict=controller_param_dict)
        self.control_input = np.zeros((12, ))

        self.I = Index(num_fps = self.num_fps)



    
    # -------------------------------------------------------------------
    def mainLoop(self):
        # the first second in unity is not stable, so we do nothing in the first second
        for k in range(10):
            state, reward, done, _ = self.env.step(self.control_input)
            state[self.I.left_end_avel_idx + self.I.right_end_avel_idx] /= 2*np.pi  # change the unit of the input angular velocity from rad/s  to 2pi*rad/s

        while rclpy.ok():
            self.control_input = self.controller.generateControlInput(state).copy()
            self.control_input[[3, 4, 5, 9, 10, 11]] *= 2*np.pi  # change the unit of the output angular velocity from 2pi*rad/s  torad/s

            state, reward, done, _ = self.env.step(self.control_input)
            state[self.I.left_end_avel_idx + self.I.right_end_avel_idx] /= 2*np.pi # change the unit of the input angular velocity from rad/s  to 2pi*rad/s

            if done: # Time up (30s), the env and the controller are reset. Next case with different desired shapes.
                self.controller.reset(state)
                state = self.env.reset()


def main(args=None):
    try:
        rclpy.init(args=args)

        env = Environment()
        env.mainLoop()

    except rclpy.exceptions.ROSInterruptException:
        print("program interrupted before completion.")

# --------------------------------------------------------------------------
if __name__ == '__main__':
    main()