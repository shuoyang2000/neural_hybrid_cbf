import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from utils.visual import make_video
from utils.frenet_utils import cartesian_to_frenet
from scripts.racing.evaluation_config import method_class

car_width = 0.31
car_length = 0.58
video_zoom_in_width = car_length * 30
video_zoom_in_height = car_length * 22

traj_dir = 'results/racing_results/'
frame_dir = "results/frames/"
video_dir = "results/racing_results/"
fig_dir = "results/racing_results/"
map_config_dir = "map_config/"

for dir in {traj_dir, frame_dir, video_dir, fig_dir, map_config_dir}:
    os.makedirs(dir, exist_ok=True)

method_display_dict = {
    "global_cbf": "Global CBF",
    "local_switch_unaware_cbf": "Switch-unaware local CBFs",
    "local_switch_aware_cbf": "Switch-aware neural local CBFs",
    "mpc": "MPC"
}

VISUAL_MODE = "video" # generate video based on trajectory
VISUAL_MODE = "map" # generate map track with different frictions
VISUAL_MODE = "map_with_traj" # generate map track with different frictions
VERBOSE = False # print infomation or not

with open(traj_dir + 'traj_' + method_class + '.json') as f:
    data_egp = json.load(f)
    keys = list(data_egp.keys())
    for key in keys:
        data_egp[key] = np.array(data_egp[key])

print(f"[Info] Starting visualization for {VISUAL_MODE}")

trajectory = np.array([data_egp['x'], data_egp['y'], data_egp['theta']])
centerline = np.load(map_config_dir + "smooth_centerline.npy")

map_resolution = 0.0625
origin_x, origin_y = -78.21853769831466, -44.37590462453829

map_matrix = np.asarray(Image.open(map_config_dir + 'example_map.png'))
boundary_point = np.where(map_matrix == 0)
boundary_point_y = ((map_matrix.shape[0] - boundary_point[0]) * map_resolution + origin_y)
boundary_point_x = (boundary_point[1] * map_resolution + origin_x)

high_friction_index, mid_friction_index, low_friction_index = [], [], []
for i in range(len(boundary_point_x)):
    frenet_pos = cartesian_to_frenet(np.array([boundary_point_x[i], boundary_point_y[i]]), centerline[:, 1:3])
    if frenet_pos[0] < 5.2 :
        mid_friction_index.append(i)
    elif frenet_pos[0] < 43.2:
        low_friction_index.append(i)
    elif frenet_pos[0] < 114.3:
        high_friction_index.append(i)
    elif frenet_pos[0] < 131.2:
        low_friction_index.append(i)
    else:
        mid_friction_index.append(i)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
low_friction_plot = plt.scatter(boundary_point_x[low_friction_index], boundary_point_y[low_friction_index], c='#17becf', s=1)
mid_friction_plot = plt.scatter(boundary_point_x[mid_friction_index], boundary_point_y[mid_friction_index], c='#1f77b4', s=1)
high_friction_plot = plt.scatter(boundary_point_x[high_friction_index], boundary_point_y[high_friction_index], c='#ff7f0e', s=1)

if VISUAL_MODE == "map" or VISUAL_MODE == "map_with_traj":
    # plt.scatter(centerline[:, 1], centerline[:, 2], s=1)
    if VISUAL_MODE == "map_with_traj":
        traj_len = trajectory.shape[1]
        sample_num = 80
        for i in range(int(traj_len / sample_num)):
            car_position = (trajectory[0, i * sample_num] - car_width / 2., trajectory[1, i * sample_num] - car_length / 2.)
            car_hearding = trajectory[2, i * sample_num] / np.pi * 180 - 90
            car = patches.Rectangle(car_position, car_width, car_length, angle=car_hearding, rotation_point = 'center', color="black",  alpha=0.50)
            ax.add_patch(car)
            if i == 0:
                init_position = car_position
        ax.text(init_position[0], init_position[1] + 5., 'Start here', fontsize = 12, color = 'red')
        ax.arrow(init_position[0] - 0.6, init_position[1] + 4., 3.4, - 0.6, width=0.2, color = 'red')

    plt.xlabel('Position x [m]', fontsize=15)
    plt.ylabel('Position y [m]', fontsize=15)
    plt.legend((low_friction_plot, mid_friction_plot, high_friction_plot),
            ('Low friction', 'Mid friction', 'High friction'),
            scatterpoints=50,
            loc='upper right',
            ncol=1,
            fontsize=15)
    image_name = 'multi_friction_track.png' if VISUAL_MODE == "map" else 'map_with_traj.png'
    plt.savefig(fig_dir + image_name)
    print(f"The multi-friction map is saved under directory {fig_dir}")

elif VISUAL_MODE == "video":
    traj_len = trajectory.shape[1]
    for i in range(traj_len):
        if i % 2 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            low_friction_plot = plt.scatter(boundary_point_x[low_friction_index], boundary_point_y[low_friction_index], c='#17becf', s=1)
            mid_friction_plot = plt.scatter(boundary_point_x[mid_friction_index], boundary_point_y[mid_friction_index], c='#1f77b4', s=1)
            high_friction_plot = plt.scatter(boundary_point_x[high_friction_index], boundary_point_y[high_friction_index], c='#ff7f0e', s=1)
            car_position = (trajectory[0, i] - car_width / 2., trajectory[1, i] - car_length / 2.)
            car_hearding = trajectory[2, i] / np.pi * 180 - 90
            car = patches.Rectangle(car_position, car_width, car_length, angle=car_hearding, rotation_point = 'center', color="black",  alpha=0.50)
            ax.add_patch(car)
            plt.figtext(0.4, 0.75, method_class, size = 14)
            plt.xlim(car_position[0] - video_zoom_in_width, car_position[0] + video_zoom_in_width)
            plt.ylim(car_position[1] - video_zoom_in_height, car_position[1] + video_zoom_in_height)
            plt.axis('off')
            plt.savefig(frame_dir + str(i // 2) + '.png')
            if VERBOSE:
                print("car_position and heading: ", car_position, car_hearding)
                print("finish frame", i)
                print("----------------------------")
            plt.close()
    make_video(frame_dir, video_dir)
    print(f"The video (new_video.mp4) for {method_class} is saved under the directory {video_dir}")

else:
    print(f"[Error] Visualization mode {VISUAL_MODE} is not defined")
    exit(-1)
