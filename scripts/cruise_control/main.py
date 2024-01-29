import jax.numpy as jnp
import numpy as np
import sys, os
import time

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
from termcolor import colored

from config import *


def render_frame(i, colorbar=False):
    # global cont_new
    global cont
    for c in cont.collections:
        c.remove()
    # timestamp.set_text("Time step = {}".format(i))

    if traj[i*20][0] >= switch_position - 2. and traj[i*20][0] <= switch_position + 2.:
        switch_index = i
        newpoint, = ax.plot(traj[i*20][1], traj[i*20][2], color="red", marker="o")
        ## for our (switch aware cbf) traj
        if method_index == 1:
            ax.text(14, 50, 'switching \n state', fontsize = 22, color = 'red')
            ax.arrow(17,48, 1.6,-1.5,width=0.2, color = 'red')
        # for switch-unaware cbf traj
        elif method_index == 0:
            ax.text(15, 49, 'switching \n state', fontsize = 22, color = 'red')
            ax.arrow(19.6,47, 1.6,-1.5,width=0.2, color = 'red')
        # for global CBF
        elif method_index == 2:
            ax.text(12, 53, 'switching \n state', fontsize = 22, color = 'red')
            ax.arrow(16,52, 1.6,-1.5,width=0.2, color = 'red')
        # for mpc
        else:
            ax.text(13, 50, 'switching \n state', fontsize = 22, color = 'red')
            ax.arrow(17, 46, 1.6,-1.5,width=0.2, color = 'red')

    else:
        newpoint, = ax.plot(traj[i*20][1], traj[i*20][2], color='blue', marker="o")
    
    # if traj[i*20][0] >= switch_position + 0.5:
    #     cont = ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[2], target_values_ice[-1][i].T, levels=[0, 200], 
    #              colors='green', alpha=0.3)
    # else:
    #     cont = ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[2], our_refine_cbf_nn_values[i].T, levels=[0, 200], 
    #              colors='green', alpha=0.3)
    cont = ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[2], target_values_ice[-1][i].T, levels=[0, 200], 
                 colors='green', alpha=0.3)

def model2cbf():

    refine_cbf_nn = load_model()
    grid_shape = grid_norm.states.shape
    assert grid_shape[-1] == 3, "Grid is not 3-dim!"

    state_array = np.asarray(grid_norm.states).copy()
    state_tensor = torch.from_numpy(state_array).reshape([grid_shape[0] * grid_shape[1] * grid_shape[2], grid_shape[3]]) # 1030301 * 3
    state_time_tensor = torch.cat((-1 * torch.ones(state_tensor.shape[0], 1), state_tensor), 1) # 1030301 * 4
    model_input = {'coords': state_time_tensor.reshape([1, state_time_tensor.shape[0], state_time_tensor.shape[1]])}
    model_out = refine_cbf_nn(model_input)['model_out']  # 1030301 * 1
    refine_cbf_nn_values = model_out.reshape([grid_shape[0], grid_shape[1], grid_shape[2]]).detach().numpy() # 101 * 101 * 101
    # save data
    with open(dataset_path + "/refine_cbf_nn_values.npy", "wb") as f:
        jnp.save(f, refine_cbf_nn_values)

    refined_cbf_dry_nn = refine_cbfs.TabularControlAffineCBF(acc_dry, grid=grid)
    refined_cbf_dry_nn.vf_table = np.array(refine_cbf_nn_values)

    return refined_cbf_dry_nn, refine_cbf_nn_values


def run_experiment(refined_cbf_dry):

    x0 = np.array([0, 30, 90])
    # x0 = np.array([60, 23, 55])
    n_sims_per_start = 1
    t_sim = 20

    desired_vel = 35
    feedback_gain = 200
    nominal_policy = lambda x, t: np.atleast_1d(np.clip(-feedback_gain * (x[..., 1] - desired_vel), umin_dry, umax_dry))

    alpha_dry = lambda x: 0.5 * x
    alpha_ice = lambda x: 5. * x

    acc_asif_dry = ControlAffineASIF(acc_dry, acc_cbf_dry, alpha=alpha_dry, nominal_policy=nominal_policy, umin=umin_dry, umax=umax_dry)
    acc_asif_ice = ControlAffineASIF(acc_ice, acc_cbf_ice, alpha=alpha_ice, nominal_policy=nominal_policy, umin=umin_ice, umax=umax_ice)
    acc_asif_global = ControlAffineASIF(acc_ice, acc_cbf_ice, alpha=alpha_ice, nominal_policy=nominal_policy, umin=umin_dry, umax=umax_dry)
    acc_asif_dry_refined = ControlAffineASIF(acc_dry, refined_cbf_dry, alpha=alpha_dry, nominal_policy=nominal_policy, umin=umin_dry, umax=umax_dry)

    experiment_paper = RolloutTrajectory('acc_example', start_x=x0, n_sims_per_start=n_sims_per_start, t_sim=t_sim)
    method_num = 3
    for method in range(method_num):

        if method == 0:
            phase1_controller = acc_asif_dry_refined
        elif method == 1:
            phase1_controller = acc_asif_dry
        elif method == 2:
            phase1_controller = acc_asif_global
        else:
            raise ValueError("Are you testing MPC? MPC should be run by mpc.py")

        _, jump_state, traj_method, control_record = experiment_paper.run_hybrid(acc_dry, acc_ice, {'nominal': nominal_policy, 'Analytical': phase1_controller}, {'nominal': nominal_policy, 'Analytical': acc_asif_ice}, switch_position)

        if method == 0:
            traj_ours = traj_method
            control_ours = control_record
        elif method == 1:
            traj_vanilla = traj_method
            control_vanilla = control_record
        elif method == 2:
            traj_global = traj_method
            control_global = control_record
        else:
            raise ValueError("Are you testing MPC? MPC should be run by mpc.py")
            
        assert abs(jump_state[0, 0] - switch_position) <= switch_radius

        return traj_vanilla, traj_global, traj_ours, control_vanilla, control_global, control_ours


if __name__ == "__main__":

    global our_refined_cbf_dry_nn, our_refine_cbf_nn_values

    if not have_reachability_data:
        print("[Info] You do not have initial local CBF data, so computing them now.")
        print("[Info] It takes around 5 mins on a desktop.")
        compute_reachability()

    our_refined_cbf_dry_nn, our_refine_cbf_nn_values = model2cbf()

    print(colored('[Info] Starting simulation.', 'blue'))

    if not have_traj:
        start_time = time.time()
        traj_vanilla, traj_global, traj_ours, control_vanilla, control_global, control_ours = run_experiment(our_refined_cbf_dry_nn)
        end_time = time.time()
        print("[Info] Sim time: ", end_time - start_time)
        # save trajectories data
        if save_traj:
            with open(dataset_path + "/traj_vanilla.npy", "wb") as f:
                jnp.save(f, traj_vanilla)
            with open(dataset_path + "/traj_global.npy", "wb") as f:
                jnp.save(f, traj_global)
            with open(dataset_path + "/control_vanilla.npy", "wb") as f:
                jnp.save(f, control_vanilla)
            with open(dataset_path + "/control_global.npy", "wb") as f:
                jnp.save(f, control_global)
            with open(dataset_path + "/traj_ours.npy", "wb") as f:
                jnp.save(f, traj_ours)
            with open(dataset_path + "/control_ours.npy", "wb") as f:
                jnp.save(f, control_ours)
        traj_mpc = jnp.load(dataset_path + "/traj_mpc.npy")
    else:
        traj_vanilla = jnp.load(dataset_path + "/traj_vanilla.npy")
        traj_global = jnp.load(dataset_path + "/traj_global.npy")
        traj_ours = jnp.load(dataset_path + "/traj_ours.npy")
        traj_mpc = jnp.load(dataset_path + "/traj_mpc.npy")
        control_vanilla = jnp.load(dataset_path + "/control_vanilla.npy")
        control_global = jnp.load(dataset_path + "/control_global.npy")
        control_ours = jnp.load(dataset_path + "/control_ours.npy")

    total_steps = traj_global.shape[0] + traj_vanilla.shape[0] + traj_ours.shape[0] - 3
    # if not have_traj:
    #     print("[Info] Average QP time: ", (end_time - start_time) / total_steps)
    target_values_dry = jnp.load(dataset_path + "/target_values_dry.npy")
    target_values_ice = jnp.load(dataset_path + "/target_values_ice.npy")
    assert target_values_dry.shape == target_values_ice.shape and target_values_dry[-1].shape == our_refine_cbf_nn_values.shape
    
    if visualization:
        print(colored('Starting visualization.', 'blue'))

        proxy = [] 
        fig, ax = plt.subplots(figsize=(9, 16))
        obstacle_viz = ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[2], obstacle_dry[0].T, levels=[-100, 0], 
                        colors='red', alpha=0.3)
        cs = ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[2], target_values_dry[-1][0].T, levels=[0, 200], 
                        colors='grey', alpha=0.3)
        cs = ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[2], target_values_ice[-1][0].T, levels=[0, 200], 
                        colors='black', alpha=0.3)
        cont = ax.contour(grid.coordinate_vectors[1], grid.coordinate_vectors[2], target_values_dry[0][0].T, levels=[0], 
                        colors=np.array([colors[3]]))   # Initial CBF
        proxy += [plt.Rectangle((0,0),1,1,ec =colors[3], fc='white', lw=5)
                for pc in cont.collections]
        proxy += [plt.Rectangle((0,0),1,1,ec =colors[-1], fc='white', lw=5) for pc in cont.collections]
        timestamp = ax.text(0.05, 0.9, "", transform=ax.transAxes)
        ax.grid()
        proxy2 = []
        proxy2 += [plt.Rectangle((0,0),1,1,fc = alt_colors[3], ec=alt_colors[3], alpha=0.3)
                for pc in cs.collections]

        proxy2 += [plt.Rectangle((0,0),1,1,fc = 'grey', ec='grey', alpha=0.3)
                for pc in cs.collections]

        proxy2 += [plt.Rectangle((0,0),1,1, fc='darkgreen', ec='darkgreen', alpha=0.3) for pc in obstacle_viz.collections]

        # proxy2 += [plt.Rectangle((0,0),1,1, fc='green', ec='green', alpha=0.3) for pc in obstacle_viz.collections]

        legend_entries = ["$\partial \mathcal{C}_h$", "$\partial \mathcal{C}_h(t)$"]
        if toggle_hjr_visualization:
            legend_entries += ["$\partial \mathcal{C}_{\ell}(t)$"]

        ax.arrow(29,92,-2,-8,width=0.2, color = 'blue')
        ax.text(27, 94, 'trajectory', fontsize = 22, color = 'blue')

        ax.legend(proxy2, ["unsafe area", "safe set $\mathcal{C}_{dry}$", "safe set $\mathcal{C}_{ice}$"],
                loc='center', bbox_to_anchor=(0.72, 0.2), ncol=1, columnspacing=1.5, handletextpad=0.4,
                facecolor=[0.8, 0.8, 0.8], edgecolor='black')

        ax.set_ylabel("Distance between vehicles ($z$) [m]")
        ax.set_xlabel("Ego velocity ($v$) [m/s]")

        global traj
        global method_index

        method_index = 1

        if method_index == 0:
            traj = traj_vanilla
        elif method_index == 1:
            traj = traj_ours
        elif method_index == 2:
            traj = traj_global
        else:
            traj = traj_mpc

        render_frame(0, False)
        animation = anim.FuncAnimation(fig, render_frame, 50, interval=200)
        video_file_name = results_path + f"/trajectory_{index_method_dict[method_index]}.gif"
        if save_video:
            animation.save(video_file_name, writer='ffmpeg')
        print(f"[Info] Finish the visualization for {index_method_dict[method_index]} and the gif has been saved.")