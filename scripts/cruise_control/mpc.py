import numpy as np
import casadi as ca
import time, os

sim_time = 20.
mpc_sample_time = 0.1
sim_sample_time = 0.01
look_ahead_time = 1.8
desired_speed = 35
horizon = 70 # almost the minimum horizon to make the whole optimization problems feasible
dataset_path = 'dataset/acc_data/'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# ACC dynamics (which is consistent with casadi)
class acc_dynamics():
    def __init__(self, params):
        self.state_dim = 3
        self.mass = params["mass"]
        self.g = params["g"]
        self.f0 = params["f0"]
        self.f1 = params["f1"]
        self.f2 = params["f2"]
        self.rolling_resistance = lambda x: self.f0 + self.f1 * x[0, 1] + self.f2 * x[0, 1] ** 2
        self.v0 = params["v0"]
        self.coefficient = params["coefficient"]
        
    def open_loop_dynamics(self, state):
        f0 = state[0, 1]
        f1 = -1 / self.mass * self.rolling_resistance(state)
        f2 = self.v0 - state[0, 1]
        f = ca.horzcat(f0, f1, f2)
        return f

    def control_matrix(self, state):
        B1 = 1 / self.mass 
        B = ca.horzcat(0., B1, 0.)
        return B

class mpc_acc():

    def __init__(self, horizon):
        self.acc_dry, self.acc_ice, self.params_dry, self.params_ice = models()
        self.horizon = horizon
        self.look_ahead_time = look_ahead_time
        self.sampling_time = mpc_sample_time
        self.desired_speed = desired_speed
        self.control_limit_dry = self.acc_dry.coefficient * self.params_dry['mass'] * self.params_dry['g']
        self.control_limit_ice = self.acc_ice.coefficient * self.params_ice['mass'] * self.params_ice['g']

    def plan(self, state, state_sequence):

        # setup
        opti = ca.Opti()

        # variables
        all_states = opti.variable(self.horizon + 1, 3)
        all_controls = opti.variable(self.horizon)

        # constraint
        dry_dynamic_index = np.where(state_sequence[:, 0] < 100.)[0] # filter our dry dynamic index
        opti.subject_to([all_states[0, :] == state]) # initial state constraint
        for i in range(self.horizon):
            # choose current dynamic
            current_dynamic = self.acc_dry if (i in dry_dynamic_index) else self.acc_ice
            # chooise current control bound
            current_limit = self.control_limit_dry if (i in dry_dynamic_index) else self.control_limit_ice 
            
            opti.subject_to(all_controls[i] <= current_limit) # control upper bound
            opti.subject_to(all_controls[i] >= - current_limit) # control lower bound

            # dynamic evolution
            opti.subject_to(all_states[i+1, :] == all_states[i, :] + self.sampling_time * 
                            (current_dynamic.open_loop_dynamics(all_states[i, :]) + 
                            current_dynamic.control_matrix(all_states[i, :]) * all_controls[i]))
            # safety constraint
            opti.subject_to(all_states[i+1, 2] - self.look_ahead_time * all_states[i+1, 1] >= 0)

        # objective
        cost1 = ca.sum1((all_states[1:, 2] - self.look_ahead_time * all_states[1:, 1]) ** 2) / 10 # be closer to the leading car
        cost2 = ca.sum1((all_states[1:, 1] - self.desired_speed) ** 2) / 100 # encourage desired speed
        obj = cost1 + cost2
        opti.minimize(obj)

        # solve
        p_opts = {"print_time": False, "verbose": False}
        s_opts = {"print_level": 0}
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()
        safe_input = sol.value(all_controls[0])
        next_state = sol.value(all_states[1, :])
        state_sequence = sol.value(all_states[1:, :]) # future states, used to foresee dynamic switching

        return safe_input, next_state, state_sequence


    '''
    1. when the mpc_sample_time is not equal to sim_sample_time, we use this function to interpolate the trajectory
    2. Reason why mpc_sample_time should be larger than sim_sample_time: higher frequency of mpc leads more horizon 
        to let the optimization problems feasible
    3. We apply the same control input while interpolating
    4. Input: current state and current control (comuted by MPC)
    5. Output: next a couple of states (all_states) by applying the current control multiple times and controls (all_controls)
    '''

    def rollout(self, state, control):
        current_state = state
        all_states = current_state
        all_controls = []
        down_sampling = int(mpc_sample_time / sim_sample_time)
        assert down_sampling == mpc_sample_time / sim_sample_time, "[Error] MPC sample time is not divisable by sim sample time"
        for i in range(down_sampling):
            current_dynamic = self.acc_dry if current_state[0, 0] < 100. else self.acc_ice # dynamic switching
            current_state = current_state + sim_sample_time * (
                current_dynamic.open_loop_dynamics(state) + current_dynamic.control_matrix(state) * control)
            all_states = np.concatenate((all_states, current_state), axis=0)
            all_controls.append(control)

        return all_states, all_controls

def models():

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
    params_dry['coefficient'] = 0.3
    acc_dry = acc_dynamics(params_dry)

    # ice road
    params_ice = params_common.copy()
    params_ice['f0'] = 0.1
    params_ice['f1'] = 5
    params_ice['f2'] = 0.25
    params_ice['coefficient'] = 0.1
    acc_ice = acc_dynamics(params_ice)

    return acc_dry, acc_ice, params_dry, params_ice

def main():
    init_state = np.array([0, 30, 90]).reshape(1, 3)
    mpc = mpc_acc(horizon)
    current_state = init_state
    state_sequence = np.tile(current_state, horizon).reshape(horizon, 3) # initialization
    trajectory = current_state
    controls = []
    mpc_used_time, current_time = 0., 0.

    # sim starts
    while current_time < sim_time:
        mpc_start_time = time.time()
        current_control, next_state, state_sequence = mpc.plan(current_state, state_sequence)
        mpc_used_time += time.time() - mpc_start_time
        current_state = next_state.reshape(1,3)
        current_time += mpc_sample_time
        if sim_sample_time < mpc_sample_time:
            all_states, all_controls = mpc.rollout(current_state, current_control)
        else:
            all_states, all_controls = current_state, current_control
        safety_value = current_state[0, 2] - look_ahead_time * current_state[0, 1]
        trajectory = np.concatenate((trajectory, all_states), axis=0)
        controls.extend(all_controls)
        print("[Info] Current state: ", current_state, "Safety value: ", safety_value)

    # save data
    with open(dataset_path + "/traj_mpc.npy", "wb") as f:
        np.save(f, trajectory)
    with open(dataset_path + "/control_mpc.npy", "wb") as f:
        np.save(f, np.array(controls))
    print("MPC average computation time: ", mpc_used_time / (sim_time / mpc_sample_time))

if __name__ == '__main__':
    main()
