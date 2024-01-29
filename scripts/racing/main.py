import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import json

from f110_gym.envs.base_classes import Integrator
from f110_gym.envs import F110Env
from utils.frenet_utils import cartesian_to_frenet
from utils.planner import PurePursuitPlanner
from scripts.racing.evaluation_config import *
from scripts.racing.mpc import mpc_f110

traj_data_dir = "results/racing_results/"
map_config_dir = "map_config/"

# setup map config and centerline
yaml_name = map_config_dir + "config_example_map"
centerline = np.load(map_config_dir + "smooth_centerline.npy")
with open(yaml_name + '.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)


def main():
    """
    main entry point
    """

    planner = PurePursuitPlanner(conf, (0.17145+0.15875))
    def render_callback(env_renderer):

        # custom extra drawing function
        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    env.add_render_callback(render_callback)
    
    obs, step_time, done, info, init_state = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    
    laptime = 0.0
    start = time.time()
    position_x, position_y, theta = [], [], []
    print("[Info] we are using " + method_class + " as our controller.")
    if method_class == 'mpc':
        current_state = np.array([init_state[0], init_state[1], 0., 0., init_state[2], 0., 0.])
        mpc_counter = 0
        mpc_frequency = 0.1
        racing_mpc = mpc_f110()

    while not done:

        if method_class in cbf_class_set:
            speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], vehicle_params['tlad'], vehicle_params['vgain'])
            obs, step_time, done, info, current_state = env.step(np.array([[steer, 0.9 * speed]]))
        
        elif method_class == 'mpc':
            if obs['linear_vels_x'][0] < 2.0: # low velocity phase, around the starting time
                sv, accl = racing_mpc.solve(current_state, obs)
                obs, step_time, done, info, current_state = env.step(np.array([[sv, accl]]))
            else: # trigger mpc
                if mpc_counter % (int(1 / mpc_frequency)) == 0:
                    sv, accl = racing_mpc.solve(current_state, obs)
                obs, step_time, done, info, current_state = env.step(np.array([[sv, accl]]))
                mpc_counter += 1
        
        else:
            raise ValueError(f"{method_class} is not implemented, please choose from " + cbf_class_set + " or mpc.")
        
        if render_video:
            env.render(mode='human')
        frenet_pos = cartesian_to_frenet(np.array([obs['poses_x'][0], obs['poses_y'][0]]), centerline[:, 1:3])

        if verbose:
            print("pose (x, y, theta): ", obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0])
            print("current_state:", current_state)
            print("frenet pose: ", frenet_pos)
            print("-------------------------")
        position_x.append(obs['poses_x'][0])
        position_y.append(obs['poses_y'][0])
        theta.append(obs['poses_theta'][0])
        laptime += step_time

        # friction updates
        env.params['mu'] = update_friction(frenet_pos[0])

    traj_data = {"x": position_x, "y": position_y, "theta": theta}
    with open(traj_data_dir + "traj_" + method_class + ".json", 'w') as f:
        json.dump(traj_data, f)
    env.close()
    print('[Info] Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()