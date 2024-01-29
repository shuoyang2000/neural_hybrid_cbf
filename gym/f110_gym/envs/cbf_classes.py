import math
import torch
import numpy as np

import f110_gym.envs.modules as modules
from utils.frenet_utils import cartesian_to_frenet
from scripts.racing.evaluation_config import *

# TODO: to write this as a base class and build a new class on this base class

if method_class in cbf_class_set:
    cbf_class = method_class 
else:
    raise ValueError(f"{method_class} is not a CBF approach.")

class wall_cbf_select():

    def __init__(self):
        self.cbf_math = wall_cbf_math()
        self.cbf_nn = wall_cbf_nn()

    def select_cbf(self, x):
        frenet_cor = cartesian_to_frenet(x[0:2], centerline[:, 1:3])
        assert cbf_class in cbf_class_set, "cbf_class is not defined"
        if cbf_class == "local_switch_aware_cbf":
            if frenet_cor[0] >= 100.0 and frenet_cor[0] < 114.3:
                return self.cbf_nn
            else:
                return self.cbf_math
        else:
            return self.cbf_math

class wall_cbf_nn():

    def __init__(self):
        self.model_left, self.model_right =  self.load_model()

    def load_model(self):
        
        model_left = modules.SingleBVPNet(in_features=8, out_features=1, type='sine', mode='mlp',
                                final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
        model_right = modules.SingleBVPNet(in_features=8, out_features=1, type='sine', mode='mlp',
                                               final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
        model_path = "models/f110_gym/"
        ckpt_path_left = model_path + 'model_left.pth'
        ckpt_path_right = model_path + 'model_right.pth'
        checkpoint_left = torch.load(ckpt_path_left, map_location=torch.device('cpu'))
        checkpoint_right = torch.load(ckpt_path_right, map_location=torch.device('cpu'))

        try:
            model_left_weights = checkpoint_left['model']
        except:
            model_left_weights = checkpoint_left
        try:
            model_right_weights = checkpoint_right['model']
        except:
            model_right_weights = checkpoint_right

        model_left.load_state_dict(model_left_weights)
        model_left.eval()
        model_right.load_state_dict(model_right_weights)
        model_right.eval()

        return model_left, model_right


    def hx_left_wall(self, x):
        return  self.hx_inner(x, self.model_left)

    def hx_right_wall(self, x):
        return  self.hx_inner(x, self.model_right)

    def hx_inner(self, x, network):
        x_new = np.copy(x)
        x_new[0:2] = cartesian_to_frenet(x_new[0:2], centerline[:, 1:3])
        x_new[0] -= 99.0
        if x_new[4] > math.pi:
            x_new[4] -= 2*math.pi

        x_new = (x_new - SCALE_MEAN) / (UPBOUND - LOWBOUND) * 2
        assert np.abs(x_new.any()) <= 1, "normalization is not correct"
        state_dim = x_new.shape[0]
        time_x = torch.cat((torch.tensor([[INITIAL_TIME]]), torch.from_numpy(x_new.reshape(1, state_dim))), -1)
        model_input = {'coords': time_x}
        model_output = network(model_input)['model_out'] * OUTPUT_FACTOR + MEAN_FACTOR

        return model_output.detach().numpy()

    def dhdx_left(self, x):

        grad = self.dhdx_inner(x, self.model_left)
        x_new = np.copy(x)
        x_new[0:2] = cartesian_to_frenet(x_new[0:2], centerline[:, 1:3])
        yaw = sp.calc_yaw(x_new[0])
        if np.sin(x_new[4]+x_new[6]-yaw)>0:
            return grad
        else:
            return np.zeros_like(grad)

    def dhdx_right(self, x):
        grad = self.dhdx_inner(x, self.model_right)
        x_new = np.copy(x)
        x_new[0:2] = cartesian_to_frenet(x_new[0:2], centerline[:, 1:3])
        yaw = sp.calc_yaw(x_new[0])
        if np.sin(x_new[4]+x_new[6]-yaw)<0:
            return grad
        else:
            return np.zeros_like(grad)

    def dhdx_inner(self, x, network):

        """ dh/dx used in Lfhx and Lghx """
        x_new = np.copy(x)
        x_new[0:2] = cartesian_to_frenet(x_new[0:2], centerline[:, 1:3])
        x_new[0] -= 99.0
        if x_new[4] > math.pi:
            x_new[4]-= 2*math.pi
        x_new = (x_new - SCALE_MEAN) / (UPBOUND - LOWBOUND) * 2
        assert np.abs(x_new.any()) <= 1, "normalization is not correct"
        state_dim = x_new.shape[0]
        time_x = torch.cat((torch.tensor([[INITIAL_TIME]]), torch.from_numpy(x_new.reshape(1, state_dim))), -1)
        time_x_grad = torch.cat((time_x, time_x, time_x, time_x, time_x, time_x, time_x), 0)
        time_x_grad = torch.cat((time_x_grad, time_x_grad), 0)
        epsilon = 0.001
        for i in range(state_dim):
            time_x_grad[i, i + 1] += epsilon
            time_x_grad[i + state_dim, i + 1] -= epsilon
        model_input = {'coords': time_x_grad}
        model_output = network(model_input)['model_out'] * OUTPUT_FACTOR + MEAN_FACTOR
        gradient = np.zeros(state_dim)
        for i in range(state_dim):
            gradient[i] = (model_output[i] - model_output[i + state_dim]) / (2 * epsilon)

        return gradient / (UPBOUND - LOWBOUND) * 2

    def alpha(self, hx):
        """ class K function """
        return 0.1 * hx


class wall_cbf_math():
    
    def __init__(self):
        self.params = cbf_params
        self.track_halfw, self.safety_margin = self.params["track_width"] / 2, self.params["wall_margin"]
        self.max_brake_lat = 100  # fixed

    def change_barke(self, x):

        frenet_pos = cartesian_to_frenet(x[0:2], centerline[:, 1:3])
        if cbf_class == "global_cbf":
            self.max_brake_lat = 0.01
        else:
            if frenet_pos[0]< 5.2 :
                self.max_brake_lat = 50.0
            elif frenet_pos[0] < 43.2:
                self.max_brake_lat = 0.01
            elif frenet_pos[0] < 114.3:
                self.max_brake_lat = 500.0
            elif frenet_pos[0] < 131.2:
                self.max_brake_lat = 0.01
            else:
                self.max_brake_lat = 50.0

    def hx_left_wall(self, x):
        frenet_pos = cartesian_to_frenet(x[0:2], centerline[:, 1:3])
        s, ey = frenet_pos[0], frenet_pos[1]
        psi_s = sp.calc_yaw(s = s) ## to check if this is correct
        dey = x[3] * np.sin(x[4] + x[6] - psi_s)

        dist_left = self.track_halfw - ey
        Dv_left = - dey  # negative if moving left
        Dv_left = min(Dv_left, 1e-10)
        hx_left = dist_left - (Dv_left**2) / (2 * self.max_brake_lat) - self.safety_margin
        return hx_left
    
    def hx_right_wall(self, x):
        frenet_pos = cartesian_to_frenet(x[0:2], centerline[:, 1:3])
        s, ey = frenet_pos[0], frenet_pos[1]
        psi_s = sp.calc_yaw(s = s) ## to check if this is correct
        dey = x[3] * np.sin(x[4] + x[6] - psi_s)
        dist_right = self.track_halfw + ey
        Dv_right = dey  # negative if moving left
        Dv_right = min(Dv_right, 1e-10)
        hx_right = dist_right - (Dv_right**2) / (2 * self.max_brake_lat) - self.safety_margin
        return hx_right
    
    def dhdx_left(self, x):
        x = x.reshape(1,7)
        multiple_x = np.concatenate((x, x, x, x, x, x, x), 0)
        multiple_x = np.concatenate((multiple_x, multiple_x), 0)
        epsilon = 0.0001
        state_dim = 7
        gradient = []
        for i in range(state_dim):
            multiple_x[i, i] += epsilon
            multiple_x[state_dim * 2 - i - 1, i] -= epsilon
            current_grad = (self.hx_left_wall(multiple_x[i]) - self.hx_left_wall(multiple_x[state_dim * 2 - i - 1])) / (2 * epsilon)
            gradient.append(current_grad)

        return gradient

    def dhdx_right(self, x):
        x = x.reshape(1,7)
        multiple_x = np.concatenate((x, x, x, x, x, x, x), 0)
        multiple_x = np.concatenate((multiple_x, multiple_x), 0)
        epsilon = 0.0001
        state_dim = 7
        gradient = []
        for i in range(state_dim):
            multiple_x[i, i] += epsilon
            multiple_x[state_dim * 2 - i - 1, i] -= epsilon
            current_grad = (self.hx_right_wall(multiple_x[i]) - self.hx_right_wall(multiple_x[state_dim * 2 - i - 1])) / (2 * epsilon)
            gradient.append(current_grad)
        
        return gradient
    
    def alpha(self, hx):
        """ class K function """
        return 1.0 * hx