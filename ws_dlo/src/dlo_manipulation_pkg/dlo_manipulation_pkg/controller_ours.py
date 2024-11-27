#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as sciR
import os
import copy

import rclpy
from rclpy.node import Node
from .RBF import JacobianPredictor
from .utils.state_index import Index

params_end_vel_max = 0.1
params_normalized_error_thres = 0.2 / 8
params_control_gain = 1.0
params_lambda_weight = 0.1
params_over_stretch_cos_angle_thres = 0.998


class Controller():
    def __init__(self, params_dict):


        # Get parameters
        self.numFPs = params_dict['num_fps']
        self.env_dim = params_dict['env_dim']
        self.env = params_dict['env']
        self.bEnableEndRotation = params_dict['bEnableEndRotation']
        self.b_left_arm = params_dict['b_left_arm'] 
        self.b_right_arm = params_dict['b_right_arm']
        self.targetFPsIdx = params_dict['targetFPsIdx'] 
        self.project_dir = params_dict['project_dir']
        self.offline_model_name = params_dict['offline_model_name']
        self.control_law = params_dict['control_law']
        self.controlRate = params_dict['controlRate']

        # Initialize other components
        self.validJacoDim = self.getValidControlInputDim(self.env_dim, self.bEnableEndRotation, self.b_left_arm, self.b_right_arm)
        self.jacobianPredictor = JacobianPredictor(params_dict=params_dict)
        self.jacobianPredictor.LoadModelWeights()

        self.I = Index(num_fps = self.numFPs)

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

        # v_init = np.zeros((J.shape[1], 1))
        v_init = np.zeros((J.shape[1], ))
        res = minimize(objectFunc, v_init, method='SLSQP', jac=jacobian, constraints=constraints_list, options={'ftol': 1e-10})
        return res.x.reshape(-1, 1)

    def generateControlInput(self, state):
        self.state_save.append(copy.deepcopy(state))

        fpsPositions = state[self.I.fps_pos_idx]
        desiredPositions = state[self.I.desired_pos_idx]

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

        # --------------------------------------------------------------------
    def validStateConstraintMatrix(self, state):
        state = np.array(state)
        left_end_pos = state[self.I.left_end_pos_idx]
        right_end_pos = state[self.I.right_end_pos_idx]

        if self.env_dim == '3D':
            fps_pos = state[self.I.fps_pos_idx].reshape(self.numFPs, 3)
            left_end_pos = state[self.I.left_end_pos_idx]
            right_end_pos = state[self.I.right_end_pos_idx]
            C1 = np.zeros((1, 12))
            C2 = np.zeros((6, 12))
        elif self.env_dim == '2D':
            fps_pos = state[self.I.fps_pos_idx].reshape(self.numFPs, 3)[:, 0:2]
            left_end_pos = state[self.I.left_end_pos_idx][0:2]
            right_end_pos = state[self.I.right_end_pos_idx][0:2]
            C1 = np.zeros((1, 6))
            C2 = np.zeros((2, 6))

        # decide whether the current state is near over-stretched
        b_over_stretch = False
        segments = fps_pos.copy()
        segments[1:, :] = (fps_pos[1:, :] - fps_pos[0:-1, :]) 
        cos_angles = np.ones((self.numFPs - 2, ))
        for i in range(2, self.numFPs - 2):
            cos_angles[i-1] = np.dot(segments[i, :], segments[i+1, :]) / (np.linalg.norm(segments[i, :]) * np.linalg.norm(segments[i+1, :]))

        ends_distance =  (right_end_pos - left_end_pos).reshape(-1, 1)
        if np.all(cos_angles > params_over_stretch_cos_angle_thres):
            b_over_stretch = True

        # calculate the C1 and C2 matrix
        if b_over_stretch:  
            pd =  ends_distance
            if self.env_dim == '3D':
                C1 = np.concatenate([-pd.T, np.zeros((1, 3)), pd.T, np.zeros((1, 3))], axis=1)
                C2_1 = np.concatenate([np.zeros((3, 3)), np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))], axis=1)
                C2_2 = np.concatenate([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)], axis=1)
                C2 = np.concatenate([C2_1, C2_2], axis=0)
            elif self.env_dim == '2D':
                C1 = np.concatenate([-pd.T, np.zeros((1, 1)), pd.T, np.zeros((1, 1))], axis=1)
                C2_1 = np.concatenate([np.zeros((1, 2)), np.eye(1), np.zeros((1, 2)), np.zeros((1, 1))], axis=1)
                C2_2 = np.concatenate([np.zeros((1, 2)), np.zeros((1, 1)), np.zeros((1, 2)), np.eye(1)], axis=1)
                C2 = np.concatenate([C2_1, C2_2], axis=0)
            return C1, C2
        else:
            return C1, C2


    # --------------------------------------------------------------------

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
        if env_dim == '2D':
            if bEnableEndRotation:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 5, 6, 7, 11]
                elif b_left_arm:
                    validJacoDim = [0, 1, 5]
                elif b_right_arm:
                    validJacoDim = [6, 7, 11]
                else:
                    validJacoDim = np.empty()
            else:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 6, 7]
                elif b_left_arm:
                    validJacoDim = [0, 1]
                elif b_right_arm:
                    validJacoDim = [6, 7]
                else:
                    validJacoDim = np.empty()
        elif env_dim == '3D':
            if bEnableEndRotation:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                elif b_left_arm:
                    validJacoDim = [0, 1, 2, 3, 4, 5]
                elif b_right_arm:
                    validJacoDim = [6, 7, 8, 9, 10, 11]
                else:
                    validJacoDim = np.empty()
            else:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 2, 6, 7, 8]
                elif b_left_arm:
                    validJacoDim = [0, 1, 2]
                elif b_right_arm:
                    validJacoDim = [6, 7, 8]
                else:
                    validJacoDim = np.empty()
        else:
            print("Error: the environment dimension must be '2D' or '3D'.")

        return validJacoDim


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
