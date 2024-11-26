#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as sciR
import os
import copy

import rclpy
from rclpy.node import Node
from RBF import JacobianPredictor
from utils.state_index import I

params_end_vel_max = 0.1
params_normalized_error_thres = 0.2 / 8
params_control_gain = 1.0
params_lambda_weight = 0.1
params_over_stretch_cos_angle_thres = 0.998


class Controller(Node):
    def __init__(self):
        super().__init__('controller_node')

        # Declare parameters
        # self.declare_parameter('DLO/num_FPs', 10)
        # self.declare_parameter('env/dimension', '3D')
        # self.declare_parameter('env/sim_or_real', 'sim')
        # self.declare_parameter('controller/enable_end_rotation', True)
        # self.declare_parameter('controller/enable_left_arm', True)
        # self.declare_parameter('controller/enable_right_arm', True)
        # self.declare_parameter('controller/object_fps_idx', [0, 1, 2])
        # self.declare_parameter('project_dir', '/path/to/project/')
        # self.declare_parameter('controller/offline_model', 'model_name')
        # self.declare_parameter('controller/control_law', 'law_name')
        # self.declare_parameter('ros_rate/env_rate', 10.0)

        # Get parameters
        self.numFPs = self.get_parameter('DLO/num_FPs').value
        self.env_dim = self.get_parameter('env/dimension').value
        self.env = self.get_parameter('env/sim_or_real').value
        self.bEnableEndRotation = self.get_parameter('controller/enable_end_rotation').value
        self.b_left_arm = self.get_parameter('controller/enable_left_arm').value
        self.b_right_arm = self.get_parameter('controller/enable_right_arm').value
        self.targetFPsIdx = self.get_parameter('controller/object_fps_idx').value
        self.project_dir = self.get_parameter('project_dir').value
        self.offline_model_name = self.get_parameter('controller/offline_model').value
        self.control_law = self.get_parameter('controller/control_law').value
        self.controlRate = self.get_parameter('ros_rate/env_rate').value

        # Initialize other components
        self.validJacoDim = self.getValidControlInputDim(self.env_dim, self.bEnableEndRotation, self.b_left_arm, self.b_right_arm)
        self.jacobianPredictor = JacobianPredictor()
        self.jacobianPredictor.LoadModelWeights()

        self.k = 0
        self.case_idx = 0
        self.state_save = []

    def normalizeTaskError(self, task_error):
        norm = np.linalg.norm(task_error)
        thres = params_normalized_error_thres * len(self.targetFPsIdx)
        if norm <= thres:
            return task_error
        else:
            return task_error / norm * thres

    def optimizeControlInput(self, fps_vel_ref, J, lambd, v_max, C1, C2):
        fps_vel_ref = fps_vel_ref.reshape(-1, 1)

        def objectFunc(v):
            v = v.reshape(-1, 1)
            cost = 0.5 * ((fps_vel_ref - J @ v).T @ (fps_vel_ref - J @ v)) + 0.5 * lambd * (v.T @ v)
            return cost[0, 0]

        def quad_inequation(v):
            v = v.reshape(-1, 1)
            return -(v.T @ v - v_max**2).reshape(-1,)

        def linear_inequation(v):
            v = v.reshape(-1, 1)
            return -(C1 @ v).reshape(-1,)

        def linear_equation(v):
            v = v.reshape(-1, 1)
            return (C2 @ v).reshape(-1,)

        def jacobian(v):
            v = v.reshape(-1, 1)
            return (-J.T @ fps_vel_ref + (J.T @ J + lambd * np.eye(J.shape[1])) @ v).reshape(-1,)

        constraints_list = [{'type': 'ineq', 'fun': quad_inequation}]
        if np.any(C1 != 0):
            constraints_list.append({'type': 'ineq', 'fun': linear_inequation})
        if np.any(C2 != 0):
            constraints_list.append({'type': 'eq', 'fun': linear_equation})

        v_init = np.zeros((J.shape[1], 1))
        res = minimize(objectFunc, v_init, method='SLSQP', jac=jacobian, constraints=constraints_list, options={'ftol': 1e-10})
        return res.x.reshape(-1, 1)

    def generateControlInput(self, state):
        self.state_save.append(copy.deepcopy(state))

        fpsPositions = state[I.fps_pos_idx]
        desiredPositions = state[I.desired_pos_idx]

        full_task_error = np.zeros((self.numFPs, 3), dtype='float32')
        full_task_error[self.targetFPsIdx, :] = np.array(fpsPositions - desiredPositions).reshape(self.numFPs, 3)[self.targetFPsIdx, :]

        target_task_error = np.zeros((3 * len(self.targetFPsIdx), 1))
        for i, targetIdx in enumerate(self.targetFPsIdx):
            target_task_error[3 * i:3 * i + 3, :] = full_task_error[targetIdx, :].reshape(3, 1)

        normalized_target_task_error = self.normalizeTaskError(target_task_error)

        Jacobian = self.jacobianPredictor.OnlineLearningAndPredictJ(state, self.normalizeTaskError(full_task_error.reshape(-1, )))

        target_J = np.zeros((3 * len(self.targetFPsIdx), len(self.validJacoDim)))
        for i, targetIdx in enumerate(self.targetFPsIdx):
            target_J[3 * i:3 * i + 3, :] = Jacobian[3 * targetIdx:3 * targetIdx + 3, self.validJacoDim]

        alpha = params_control_gain
        fps_vel_ref = -alpha * normalized_target_task_error

        lambd = params_lambda_weight * np.linalg.norm(normalized_target_task_error)
        v_max = params_end_vel_max

        C1, C2 = self.validStateConstraintMatrix(state)

        u = self.optimizeControlInput(fps_vel_ref, target_J, lambd, v_max, C1, C2)

        u_12DoF = np.zeros((12,))
        u_12DoF[self.validJacoDim] = u.reshape(-1, )

        if np.linalg.norm(u_12DoF) > v_max:
            u_12DoF = u_12DoF / np.linalg.norm(u_12DoF) * v_max

        self.k += 1

        return u_12DoF

    def reset(self, state):
        self.state_save.append(state)

        result_dir = os.path.join(self.project_dir, "results", self.env, "control", self.control_law, self.env_dim)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        np.save(os.path.join(result_dir, f"state_{self.case_idx}.npy"), self.state_save)

        self.case_idx += 1
        self.state_save = []
        self.k = 0

        if self.case_idx == 100:
            self.get_logger().info("Finished all cases, shutting down.")
            rclpy.shutdown()

        self.jacobianPredictor.LoadModelWeights()

    def getValidControlInputDim(self, env_dim, bEnableEndRotation, b_left_arm, b_right_arm):
        # Similar logic to original, omitted for brevity
        pass


def main(args=None):
    rclpy.init(args=args)
    controller = Controller()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
