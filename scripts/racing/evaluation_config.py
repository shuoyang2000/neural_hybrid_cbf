import numpy as np
from utils.cubic_spline import CubicSpline2D

# choose the method you wanna test here

# method_class = "global_cbf"
# method_class = "local_switch_unaware_cbf"
method_class = "local_switch_aware_cbf"
# method_class = "mpc"

cbf_class_set = {"local_switch_aware_cbf", "global_cbf", "local_switch_unaware_cbf"}

verbose = False # print infomation or not
render_video = True # display video while running or not

CBF_FILTER = False
if method_class in cbf_class_set:
    CBF_FILTER = True

MEAN_FACTOR = -20.0
OUTPUT_FACTOR = 20.0
INITIAL_TIME = -0.7
LOWBOUND = np.array([0.0, -1.0, -0.21, 2.5, -1.5, -5.0, -0.2])
UPBOUND = np.array([16.0, 1.0, 0.21, 7.5, 1.5, 5.0, 0.2])
SCALE_MEAN = (UPBOUND + LOWBOUND) / 2.
cbf_params = {'track_width': 3.00, 'wall_margin': 0.58,}
vehicle_params = {'C_Sf': 4.718, 'C_Sr': 5.456, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 
                  'm': 3.74, 'I': 0.04712, 'g': 9.807, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889, 'friction': 1.0489}
frictions = {'low': 0.12, 'mid': 0.6, 'high': 1.04}
centerline = np.load("map_config/smooth_centerline.npy")
x = list(centerline[:, 1])
y = list(centerline[:, 2])
sp = CubicSpline2D(x, y)

def update_friction(frenet_x):
    if frenet_x < 5.2:
        mu = frictions['mid']
    elif frenet_x < 43.2: 
        mu = frictions['low']
    elif frenet_x < 114.3:
        mu = frictions['high']
    elif frenet_x < 131.2:
        mu = frictions['low']
    else:
        mu = frictions['mid']
    return mu