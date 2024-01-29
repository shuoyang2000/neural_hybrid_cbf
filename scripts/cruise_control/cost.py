import numpy as np

dataset_path = 'dataset/acc_data/'

def traj_cost(traj: np.ndarray):
    
    assert traj.shape[1] == 3, "trajectory is not 3-dim"

    look_ahead_time = 1.8
    desired_speed = 35.
    time_step = 0.01
    cost1 = time_step * np.sum((traj[:, 2] - traj[:, 1] * look_ahead_time) ** 2) / 10
    cost2 = time_step * np.sum((traj[:, 1] - desired_speed) ** 2) / 100
    cost = cost1 + cost2

    return cost

def main():
    traj_vanilla = np.load(dataset_path + "/traj_vanilla.npy")
    traj_global = np.load(dataset_path + "/traj_global.npy")
    traj_ours = np.load(dataset_path + "/traj_ours.npy")
    traj_literature = np.load(dataset_path + "/traj_switch_aware_literature.npy")
    traj_mpc = np.load(dataset_path + "/traj_mpc.npy")
    cost_vaniall = traj_cost(traj_vanilla)
    cost_global = traj_cost(traj_global)
    cost_ours = traj_cost(traj_ours)
    cost_literature = traj_cost(traj_literature)
    cost_mpc = traj_cost(traj_mpc)
    print("Cost of switch-unaware local CBF: ", cost_vaniall)
    print("Cost of global CBF: ", cost_global)
    print("Cost of switching-aware local CBF (our approach): ", cost_ours)
    print("Cost of switching-aware local CBF (literature): ", cost_literature)
    print("Cost of MPC: ", cost_mpc)
 

if __name__ == '__main__':
    main()