import hj_reachability as hj
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(1, 'lib')
import refine_cbfs
import experiment_wrapper
import cbf_opt
from experiment_wrapper import RolloutTrajectory, TimeSeriesExperiment, StateSpaceExperiment
from cbf_opt import ControlAffineDynamics, ControlAffineCBF, ControlAffineASIF

from utils.cruise_control import ACCCBF, ACCDynamics, ACCJNPDynamics

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from utils.modules import SingleBVPNet

have_reachability_data = True # already have reachbility data or not
have_new_cbf_data = True
have_traj = True
visualization = True
save_traj = True

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': '28',
    'text.usetex': False,
    'pgf.rcfonts': False,
})

colors = sns.color_palette("tab10")
chosen_colors = [(0.5, 0.5, 0.5)]
chosen_colors.append(colors[0])
chosen_colors.append(colors[1])

colors = [(0.3, 0.3, 0.3)]
colors += [(sns.color_palette("RdYlGn_r", 7)[0])]
colors += [(sns.color_palette("RdYlGn_r", 9)[6])]
colors += [(sns.color_palette("RdYlGn_r", 9)[8])]
colors += [(4 / 255, 101 / 255, 4 / 255)]
colors = np.array(colors)

alt_colors = sns.color_palette("pastel", 9).as_hex()

index_method_dict = {
    0: "vanilla",
    1: "ours",
    2: "global",
    3: "mpc"
}

params = {'axes.labelsize': 28,'axes.titlesize':28, 'font.size': 28, 'legend.fontsize': 28, 
          'xtick.labelsize': 28, 'ytick.labelsize': 28, 'lines.linewidth': 5}
matplotlib.rcParams.update(params)

results_path = "results/acc_results"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

colors = sns.color_palette("tab10")
alt_colors = sns.color_palette("pastel", 9).as_hex()
toggle_hjr_visualization = False
save_video = True

### common parameters
params_common = dict()
params_common["dt"] = 0.01
params_common["g"] = 9.81
params_common['v0'] = 14  # lead vehicle velocity
params_common['mass'] = 1650  # vehicle mass

# dry road
params_dry = params_common.copy()
params_dry['f0'] = 0.1 * 3  # friction coefficient
params_dry['f1'] = 5 * 3  # friction coefficient
params_dry['f2'] = 0.25 * 3  # friction coefficient
acc_dry = ACCDynamics(params_dry)
acc_jnp_dry = ACCJNPDynamics(params_dry)

# ice road
params_ice = params_common.copy()
params_ice['f0'] = 0.1
params_ice['f1'] = 5
params_ice['f2'] = 0.25
acc_ice = ACCDynamics(params_ice)
acc_jnp_ice = ACCJNPDynamics(params_ice)

cbf_params = dict()
cbf_params["Th"] = 1.8
control_limit_dry = 0.3
control_limit_ice = 0.1

acc_cbf_dry = ACCCBF(acc_dry, cbf_params, control_limit_dry)
acc_jnp_cbf_dry = ACCCBF(acc_jnp_dry, cbf_params, control_limit_dry)

umax_dry = np.array([control_limit_dry * params_dry['mass'] * params_dry['g']])
umin_dry = - umax_dry

acc_cbf_ice = ACCCBF(acc_ice, cbf_params, control_limit_ice)
acc_jnp_cbf_ice = ACCCBF(acc_jnp_ice, cbf_params, control_limit_ice)

umax_ice = np.array([control_limit_ice * params_ice['mass'] * params_ice['g']])
umin_ice = - umax_ice

dyn_reachability_jnp_dry = refine_cbfs.HJControlAffineDynamics(acc_jnp_dry, control_space=hj.sets.Box(jnp.array(umin_dry), jnp.array(umax_dry)))
dyn_reachability_dry = refine_cbfs.HJControlAffineDynamics(acc_dry, control_space=hj.sets.Box(jnp.array(umin_dry), jnp.array(umax_dry)))

dyn_reachability_jnp_ice = refine_cbfs.HJControlAffineDynamics(acc_jnp_ice, control_space=hj.sets.Box(jnp.array(umin_ice), jnp.array(umax_ice)))
dyn_reachability_ice = refine_cbfs.HJControlAffineDynamics(acc_ice, control_space=hj.sets.Box(jnp.array(umin_ice), jnp.array(umax_ice)))

low_bound = jnp.array([0., 10., 0.])
high_bound = jnp.array([500, 40., 100.])
low_bound_norm = jnp.array([-1., -1., -1.])
high_bound_norm = jnp.array([1., 1., 1.])
nbr_pts = (101, 101, 101)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(low_bound, high_bound), nbr_pts)
grid_np = refine_cbfs.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(low_bound, high_bound), nbr_pts)
grid_norm = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(low_bound_norm, high_bound_norm), nbr_pts)
grid_norm_np = refine_cbfs.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(low_bound_norm, high_bound_norm), nbr_pts)

obstacle_dry = grid.states[..., 2] - acc_cbf_dry.Th * grid.states[..., 1]
obstacle_ice = grid.states[..., 2] - acc_cbf_ice.Th * grid.states[..., 1]
assert obstacle_dry.all() == obstacle_ice.all()

acc_tabular_cbf_dry = refine_cbfs.TabularControlAffineCBF(acc_dry, dict(), grid=grid)
acc_jnp_tabular_cbf_dry = refine_cbfs.TabularControlAffineCBF(acc_jnp_dry, dict(), grid=grid)
acc_tabular_cbf_dry.tabularize_cbf(acc_cbf_dry)
acc_jnp_tabular_cbf_dry.tabularize_cbf(acc_jnp_cbf_dry)

acc_tabular_cbf_ice = refine_cbfs.TabularControlAffineCBF(acc_ice, dict(), grid=grid)
acc_jnp_tabular_cbf_ice = refine_cbfs.TabularControlAffineCBF(acc_jnp_ice, dict(), grid=grid)
acc_tabular_cbf_ice.tabularize_cbf(acc_cbf_ice)
acc_jnp_tabular_cbf_ice.tabularize_cbf(acc_jnp_cbf_ice)

switch_position = 100
switch_radius = 1.
switch = abs(grid.states[..., 0] - switch_position) - switch_radius # considered as switching area

dataset_path = "dataset/acc_data"
if not os.path.isdir(dataset_path):
    os.makedirs(dataset_path)

def compute_new_cbf(target_values_dry, target_values_ice):

    # the intersection of safe dry and unsafe ice regions
    unsafe_ice_values = target_values_dry * target_values_ice

    # safe switching area
    safe_switch = jnp.maximum(switch, unsafe_ice_values[-1])

    new_obstacle = jnp.minimum(safe_switch, target_values_dry[-1])
    solver_settings = hj.SolverSettings.with_accuracy("medium",
                                                    value_postprocessor=backwards_reachable_tube(new_obstacle))
    init_value_refine_dry = refined_dry_cbf.vf_table

    target_values_refine_dry = hj.solve(solver_settings, dyn_reachability_jnp_dry, grid, times, init_value_refine_dry)
    
    # save data
    with open(dataset_path + "/target_values_refined_dry.npy", "wb") as f:
        jnp.save(f, target_values_refine_dry)
    print("finish target_values_refined_dry")
    
    return target_values_refine_dry


def refine_cbf():

    if not have_reachability_data:
        compute_reachability()

    target_values_ice = jnp.load(dataset_path + "/target_values_ice.npy")
    target_values_hjr_ice = jnp.load(dataset_path + "/target_values_hjr_ice.npy")
    target_values_dry = jnp.load(dataset_path + "/target_values_dry.npy")
    target_values_hjr_dry = jnp.load(dataset_path + "/target_values_hjr_dry.npy")

    assert jnp.sum(jnp.where(target_values_ice[-1] > 0, 1, 0) ) == jnp.sum(jnp.where(target_values_dry[-1] > 0, 1, 0) * jnp.where(target_values_ice[-1] > 0, 1, 0))

    refined_dry_cbf = refine_cbfs.TabularControlAffineCBF(acc_dry, grid=grid_np)
    refined_dry_cbf_jnp = refine_cbfs.TabularControlAffineCBF(acc_jnp_dry, grid=grid)
    print(grid_np.shape)
    print(target_values_dry[-1].shape)
    refined_dry_cbf.vf_table = np.array(target_values_dry[-1])
    refined_dry_cbf_jnp.vf_table = target_values_dry[-1]
    
    global target_values_refine_dry
    if not have_new_cbf_data:
        target_values_refine_dry = compute_new_cbf(target_values_dry, target_values_ice)
    else:
        target_values_refine_dry = jnp.load(dataset_path + "/target_values_refined_dry.npy")
    
    global refined_cbf_dry

    refined_cbf_dry = refine_cbfs.TabularControlAffineCBF(acc_dry, grid=grid_np)
    refined_cbf_dry.vf_table = np.array(target_values_refine_dry[-1])

    return refined_cbf_dry

def load_model():

    model = SingleBVPNet(in_features=4, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3)

    ckpt_path = 'models/cruise_control/model_final.pth'
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    model.eval()

    return model


def compute_reachability():

    time = 0.
    target_time = - 20.0
    times = jnp.linspace(time, target_time, 101)
    backwards_reachable_tube = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))

    solver_settings = hj.SolverSettings.with_accuracy("medium",
                                                  value_postprocessor=backwards_reachable_tube(obstacle_ice))
    init_value = acc_jnp_tabular_cbf_ice.vf_table

    # unsafe set: the last value is negative
    target_values_hjr_ice = hj.solve(solver_settings, dyn_reachability_jnp_ice, grid, times, obstacle_ice)
    target_values_ice = hj.solve(solver_settings, dyn_reachability_jnp_ice, grid, times, init_value)


    solver_settings = hj.SolverSettings.with_accuracy("medium",
                                                  value_postprocessor=backwards_reachable_tube(obstacle_dry))
    init_value_dry = acc_jnp_tabular_cbf_dry.vf_table
    target_values_hjr_dry= hj.solve(solver_settings, dyn_reachability_jnp_dry, grid, times, obstacle_dry)
    target_values_dry = hj.solve(solver_settings, dyn_reachability_jnp_dry, grid, times, init_value_dry)


    # save data

    with open(dataset_path + "/target_values_ice.npy", "wb") as f:
        jnp.save(f, target_values_ice)
    print("finish target_values_ice")
    with open(dataset_path + "/target_values_hjr_ice.npy", "wb") as f:
        jnp.save(f, target_values_hjr_ice)
    print("finish target_values_hjr_ice")
    with open(dataset_path + "/target_values_dry.npy", "wb") as f:
        jnp.save(f, target_values_dry)
    print("finish target_values_dry")
    with open(dataset_path + "/target_values_hjr_dry.npy", "wb") as f:
        jnp.save(f, target_values_hjr_dry)
    print("finish target_values_hjr_dry")
