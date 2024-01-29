import casadi
from f110_gym.envs.cbf_classes import wall_cbf_select
import numpy as np

cbf_select = wall_cbf_select()

def f1tenth_cbf_project(x, u, fx, gx, sv_min, sv_max, s_min, s_max, a_max):
    
    cbf = cbf_select.select_cbf(x)
    new_x = np.copy(x)
    cbf_select.cbf_math.change_barke(new_x)
    hx_left = cbf.hx_left_wall(new_x)
    hx_right = cbf.hx_right_wall(new_x)
    dhdx_left = cbf.dhdx_left(new_x)
    dhdx_right = cbf.dhdx_right(new_x)
    Lfh_left = dhdx_left @ fx
    Lgh_left = dhdx_left @ gx
    Lfh_right = dhdx_right @ fx
    Lgh_right = dhdx_right @ gx

    # if True: # Lfh_right + Lgh_right.dot(u) < - cbf.alpha(hx_right) or Lfh_left + Lgh_left.dot(u) < - cbf.alpha(hx_left):
    opti = casadi.Opti()

    # decision variables: control input u_safe, and CBF slack varaibles wall_slack
    u_safe = opti.variable(2)
    wall_slack = opti.variable(2)
    
    sv_min_new, sv_max_new = sv_min, sv_max
    if x[2] <= s_min:
        sv_min_new = 0.
    if x[2] >= s_max:
        sv_max_new = 0.

    # constraints
    opti.subject_to(np.array([sv_min_new, -a_max]) <= u_safe)
    opti.subject_to(u_safe <= np.array([sv_max_new, a_max]))
    opti.subject_to(Lfh_right + Lgh_right[0] * u_safe[0] + Lgh_right[1] * u_safe[1] + wall_slack[0] >= - cbf.alpha(hx_right))
    opti.subject_to(Lfh_left + Lgh_left[0] * u_safe[0] + Lgh_left[1] * u_safe[1] + wall_slack[1] >= - cbf.alpha(hx_left))
    opti.subject_to(wall_slack[0] >= 0)
    opti.subject_to(wall_slack[1] >= 0)

    # objective
    a_scale, beta_scale = 100.0 / a_max, 1.0 / sv_max
    obj = (
        a_scale * (u_safe[0] - u[0]) ** 2
        + beta_scale * (u_safe[1] - u[1]) ** 2
        + 1000 * wall_slack[0] + 1000*wall_slack[1]
    )
    opti.minimize(obj)

    # solve
    p_opts = {"print_time": False, "verbose": False}
    s_opts = {"print_level": 0}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()
    safe_input = sol.value(u_safe)

    return safe_input