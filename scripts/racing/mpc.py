import time

import numpy as np
import casadi as ca
from scripts.racing.evaluation_config import vehicle_params, frictions
from utils.frenet_utils import nearest_point

map_config_dir = "map_config/"
waypoint = np.load(map_config_dir + "waypoint_table.npy")
waypoint = [waypoint[:, 0], waypoint[:, 1], waypoint[:, 2], waypoint[:, 3], waypoint[:, 4]] # x, y, yaw, desired_velocity, frenet coordinate "s"

horizon = 20 # MPC horizon
mpc_sample_time = 0.1 # MPC step time
state_num = 7  # state number of racing car
waypoint_dist = 0.03  # distance of the adjacent element in waypoint

## MPC_objective parameter
Q_param = np.array([[20.0, 20.0, 0.0, 0.7, 8.0, 0.0, 0.0]])  # objective parameter for state in stage cost from 1 to T/2-1
Qf_param = np.array([[8.0, 8.0, 0.0, 0.7, 2.0, 0.0, 0.0]])   # objective parameter for state in stage cost at T/2
R_param = [3.0, 0.3]  # objective parameters for control input (steer and accl)
Q_param_long = np.array([[1.0, 1.0, 0.0, 0.035, 0.1, 0.0, 0.0]])  # objective parameter for state in stage cost from T/2 to T
R_param_long = [0.075, 0.015]  # objective parameters for control input (steer and accl)

mpc_time = 0.
mpc_count = 0


def calc_ref_trajectory(state, cx, cy, cyaw, sp, cs):
    """
    calc referent trajectory ref_traj in T steps: [x, y, v, yaw] and its frenet s
    using the current velocity, calc the T points along the reference path
    :param cx: Course X-Position
    :param cy: Course y-Position
    :param cyaw: Course Heading
    :param sp: speed profile
    :param ss: frenet coordinate s
    :dl: distance step
    :pind: Setpoint Index
    :return: reference trajectory ref_traj, reference steering angle
    """

    # Create placeholder Arrays for the reference trajectory for T steps
    ref_traj = np.zeros((state_num, horizon + 1))
    ncourse = len(cx)

    # Find nearest index/setpoint from where the trajectories are calculated
    _, _, _, ind = nearest_point(np.array([state[0], state[1]]), np.array([cx, cy]).T)

    # Load the initial parameters from the setpoint into the trajectory
    ref_traj[0, 0] = cx[ind]
    ref_traj[1, 0] = cy[ind]
    ref_traj[3, 0] = sp[ind]
    ref_traj[4, 0] = cyaw[ind]

    # based on current velocity, distance traveled on the ref line between time steps
    travel = abs(state[3]) * mpc_sample_time
    dind = travel / waypoint_dist
    ind_list = int(ind) + np.insert(
        np.cumsum(np.repeat(dind, horizon)), 0, 0
    ).astype(int)


    ind_list[ind_list >= ncourse] -= ncourse

    ref_traj[0, :] = cx[ind_list]
    ref_traj[1, :] = cy[ind_list]
    ref_traj[3, :] = sp[ind_list]
    cyaw[cyaw - state[4] > 5] = cyaw[cyaw - state[4] > 5] - (2 * np.pi)
    cyaw[cyaw - state[4] < -5] = cyaw[cyaw - state[4] < -5] + (2 * np.pi)
    ref_traj[4, :] = cyaw[ind_list]
    state_sequnce = cs[ind_list]

    return ref_traj, state_sequnce

## dynamic of racing car
class f110_dynamic():
    def __init__(self, para):
        self.state_dim = 7
        self.m = para['m']
        self.I = para['I']
        self.lf = para['lf']
        self.lr = para['lr']
        self.C_Sf = para['C_Sf']
        self.C_Sr = para['C_Sr']
        self.g = para['g']
        self.h = para['h']
        self.friction = para['friction']
        self.epsilon = 0.01
        self.fx_6_f = lambda x: -self.friction*self.m/((x[0,3]+self.epsilon)*self.I*(self.lr+self.lf))*(self.lf**2*self.C_Sf*(self.g*self.lr) + self.lr**2*self.C_Sr*(self.g*self.lf))*x[0,5] \
                +self.friction*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(self.g*self.lf) - self.lf*self.C_Sf*(self.g*self.lr))*x[0,6] \
                +self.friction*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(self.g*self.lr)*x[0,2]

        self.fx_6_g = lambda x: -self.friction*self.m/((x[0,3]+self.epsilon)*self.I*(self.lr+self.lf))*(self.lf**2*self.C_Sf*(-self.h) + self.lr**2*self.C_Sr*(self.h))*x[0,5] \
                +self.friction*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(self.h) - self.lf*self.C_Sf*(-self.h))*x[0,6] \
                +self.friction*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(-self.h)*x[0,2]

        self.fx_7_f = lambda x: (self.friction/((x[0,3]+self.epsilon)**2*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf)*self.lr - self.C_Sf*(self.g*self.lr)*self.lf)-1)*x[0,5] \
                -self.friction/((x[0,3]+self.epsilon)*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf) + self.C_Sf*(self.g*self.lr))*x[0,6] \
                +self.friction/((x[0,3]+self.epsilon)*(self.lr+self.lf))*(self.C_Sf*(self.g*self.lr))*x[0,2]

        self.fx_7_g = lambda x: (self.friction/((x[0,3]+self.epsilon)**2*(self.lr+self.lf))*(self.C_Sr*(self.h)*self.lr - self.C_Sf*(-self.h)*self.lf))*x[0,5]\
                -self.friction/((x[0,3]+self.epsilon)*(self.lr+self.lf))*(self.C_Sr*(self.h) + self.C_Sf*(-self.h))*x[0,6] \
                +self.friction/((x[0,3]+self.epsilon)*(self.lr+self.lf))*(self.C_Sf*(-self.h))*x[0,2]

    def open_loop_dynamics(self, state):

        f0 = state[0, 3]*ca.cos(state[0, 6]+state[0, 4])
        f1 = state[0, 3]*ca.sin(state[0, 6]+state[0, 4])
        f4 = state[0, 5]
        f5 = self.fx_6_f(state)
        f6 = self.fx_7_f(state)
        f = ca.horzcat(f0, f1, 0.0, 0.0, f4, f5, f6)
        return f


    def control_matrix_accl(self, state):

        g5 = self.fx_6_g(state)
        g6 = self.fx_7_g(state)
        g = ca.horzcat(0., 0., 0., 1., 0., g5, g6)

        return g

    def control_matrix_steer(self, state):

        return ca.horzcat(0., 0., 1., 0., 0., 0., 0.)


## function to load racing car model for different friction
def models():

    vehicle_params_high = vehicle_params.copy()
    vehicle_params_high['friction'] = frictions['high']
    f110_dynamic_high = f110_dynamic(vehicle_params_high)

    vehicle_params_middle = vehicle_params.copy()
    vehicle_params_middle['friction'] = frictions['mid']
    f110_dynamic_middle = f110_dynamic(vehicle_params_middle)

    vehicle_params_low = vehicle_params.copy()
    vehicle_params_low['friction'] = frictions['low']
    f110_dynamic_low = f110_dynamic(vehicle_params_low)

    return f110_dynamic_high, f110_dynamic_middle, f110_dynamic_low

# nonlinear mpc for f1/10 single-track model
class mpc_f110():

    def __init__(self):
        self.horizon = horizon
        self.sampling_time = mpc_sample_time
        self.high, self.middle, self.low = models()
        self.accl_limit = 9.5
        self.steer_limit = 3.2
        self.Q_param = Q_param
        self.Qf_param = Qf_param
        self.R_param = R_param
        self.Q_param_long = Q_param_long
        self.R_param_long = R_param_long
        self.infeasibility_count = 0

    def plan(self, state: np.array):
        # state is array with shape (7,)

        # compute reference trajectory; state sequence used to determine which friction dynamic to use
        ref_trajectory, state_sequence = calc_ref_trajectory(state, waypoint[0], waypoint[1], waypoint[2], waypoint[3], waypoint[4])

        # setup
        opti = ca.Opti()

        # variables
        all_states = opti.variable(self.horizon + 1, state_num)
        all_controls_accl = opti.variable(self.horizon)
        all_controls_steer = opti.variable(self.horizon)

        ## constraints

        # decide to use which friction dynamic
        high_dynamic_index_left = np.where(state_sequence > 43.2)[0]
        high_dynamic_index_right = np.where(state_sequence < 114.3)[0]
        high_dynamic_index = [x for x in high_dynamic_index_left if x in high_dynamic_index_right]

        middle_dynamic_index_left = list(np.where(state_sequence < 5.2)[0])
        middle_dynamic_index_right = list(np.where(state_sequence > 131.2)[0])
        middle_dynamic_index = middle_dynamic_index_left+middle_dynamic_index_right

        low_other = high_dynamic_index + middle_dynamic_index
        low_dynamic_index = list(set([i for i in range(self.horizon)])-set(low_other))

        # initial state constraint
        opti.subject_to([all_states[0, :] == state[None, :]])
        
        # control bound constraint
        opti.subject_to(all_controls_accl <= np.ones(self.horizon) * self.accl_limit)  # control upper bound
        opti.subject_to(all_controls_accl >= - np.ones(self.horizon) * self.accl_limit)  # control lower bound
        opti.subject_to(all_controls_steer <= np.ones(self.horizon) * self.steer_limit)  # control upper bound
        opti.subject_to(all_controls_steer >= - np.ones(self.horizon) * self.steer_limit)  # control lower bound

        for i in range(self.horizon):
            # choose current dynamic
            if i in low_dynamic_index:
                current_dynamic = self.low
            elif i in middle_dynamic_index:
                current_dynamic = self.middle
            else:
                current_dynamic = self.high

            #dynamic evolution
            opti.subject_to(all_states[i + 1, :] == all_states[i, :] + self.sampling_time *
                            (current_dynamic.open_loop_dynamics(all_states[i, :]) +
                             current_dynamic.control_matrix_accl(all_states[i, :]) * all_controls_accl[i] +
                             current_dynamic.control_matrix_steer(all_states[i, :]) * all_controls_steer[i]))



        # objective
        obj = ca.dot((all_states[self.horizon//2, :] - ref_trajectory.T[self.horizon//2, None, :]) ** 2, self.Qf_param)

        for i in range(self.horizon//2):
            obj += ca.dot((all_states[i, :] - ref_trajectory.T[i, None, :])**2, self.Q_param)
            obj += self.R_param[0]*all_controls_steer[i]**2 + self.R_param[1]*all_controls_accl[i]**2

        for i in range(self.horizon//2+1,self.horizon,1):
            obj += ca.dot((all_states[i, :] - ref_trajectory.T[i, None, :])**2, self.Q_param_long)
            obj += self.R_param_long[0] * all_controls_steer[i] ** 2 + self.R_param_long[1] * all_controls_accl[i] ** 2

        opti.minimize(obj)

        # solve
        options = {'ipopt.max_iter': 3000, 'ipopt.print_level': 0, 'verbose': False, 'print_time': False}
        opti.solver("ipopt", options)
        global mpc_count, mpc_time
        start = time.time()
        sol = opti.solve()
        end = time.time()
        mpc_time = mpc_time + end - start
        mpc_count += 1
        print("average_time:", mpc_time/mpc_count)

        safe_input_accl = sol.value(all_controls_accl)
        safe_input_steer = sol.value(all_controls_steer)

        return safe_input_steer, safe_input_accl

    def solve(self, state: np.array, obs: dict):
        # when the velocity is very low (happens only during the starting second)
        if obs['linear_vels_x'][0] < 2.0:
            accl, sv = 9.5, 0.0
            return sv, accl
        # normally, we use mpc to plan ahead and return the first action
        else:
            try:
                self.sv_list, self.accl_list = self.plan(state)
                sv = self.sv_list[0]
                accl = self.accl_list[0]
                self.infeasibility_count = 0
            except:
                # if the MPC is infeasible, we return the previously planned actions, this happens rarely.
                # print("[Warning] Current mpc problem is infeasible, using previously planned actions.")
                self.infeasibility_count += 1
                sv, accl = self.sv_list[self.infeasibility_count], self.accl_list[self.infeasibility_count]
            return sv, accl

