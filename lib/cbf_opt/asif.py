import abc
import cvxpy as cp
import casadi as ca
import numpy as np
from cbf_opt import Dynamics, ControlAffineDynamics
from cbf_opt import CBF, ControlAffineCBF, ImplicitCBF, ControlAffineImplicitCBF, BackupController
from typing import Dict, Optional
from cbf_opt.tests import test_asif

import logging


logger = logging.getLogger(__name__)

# Array = np.ndarray or torch.tensor
Array = np.ndarray
# batched_ncbf = lambda x, y: torch.bmm(x, y)
batched_cbf = lambda x, y: np.einsum("ijk,ikl->ijl", x, y)
single_cbf = lambda x, y: x @ y


class ASIF(metaclass=abc.ABCMeta):
    def __init__(self, dynamics: Dynamics, cbf: CBF, test: bool = True, **kwargs) -> None:
        self.dynamics = dynamics
        self.cbf = cbf
        self.nominal_control = None
        self.alpha = kwargs.get("alpha", lambda x: x)
        self.verbose = kwargs.get("verbose", False)
        self.solver = kwargs.get("solver", "OSQP")
        self.nominal_policy = kwargs.get("nominal_policy", lambda x, t: np.zeros(self.dynamics.control_dims))
        self.controller_dt = kwargs.get("controller_dt", self.dynamics.dt)
        if test:
            test_asif.test_asif(self)

    def set_nominal_control(self, state: Array, time: float = 0.0, nominal_control: Optional[Array] = None) -> None:
        if nominal_control is not None:
            assert isinstance(nominal_control, Array) and nominal_control.shape[-1] == (
                self.dynamics.control_dims,
            )  # TODO: can we just get  rid of this?
            self.nominal_control = nominal_control
        else:
            self.nominal_control = self.nominal_policy(state, time)

    @abc.abstractmethod
    def __call__(self, state: Array, time: float = 0.0, nominal_control: Optional[Array] = None) -> Array:
        """Implements the active safety invariance filter"""

    def save_info(self, state: Array, control: Array, time: float = 0.0) -> Dict:
        return {"unsafe": self.cbf.is_unsafe(state, time)}

    def save_measurements(self, state: Array, control: Array, time: float = 0.0) -> Dict:
        dict = (
            self.nominal_policy.save_measurements(state, control, time)
            if hasattr(self.nominal_policy, "save_measurements")
            else {}
        )
        dict["vf"] = self.cbf.vf(state, time)
        return dict

# original implementation through cvxpy, which is faster than casadi; 
# if you want to test the time performance, please use the ControlAffineASIF class of casadi version
    
class ControlAffineASIF(ASIF):
    def __init__(self, dynamics: ControlAffineDynamics, cbf: ControlAffineCBF, test: bool = True, **kwargs) -> None:
        super().__init__(dynamics, cbf, test, **kwargs)
        self.filtered_control = cp.Variable(self.dynamics.control_dims)
        self.nominal_control_cp = cp.Parameter(self.dynamics.control_dims)

        self.umin = kwargs.get("umin")
        self.umax = kwargs.get("umax")
        self.b = cp.Parameter((1,))
        self.A = cp.Parameter((1, self.dynamics.control_dims))

        self.opt_sol = np.zeros(self.filtered_control.shape)

        if test:
            test_asif.test_control_affine_asif(self)

    def setup_optimization_problem(self):
        """
        min || u - u_des ||^2
        s.t. A @ u + b >= 0
        """
        self.obj = cp.Minimize(
            cp.quad_form(self.filtered_control - self.nominal_control_cp, np.eye(self.dynamics.control_dims))
        )
        self.constraints = [self.A @ self.filtered_control + self.b >= 0]
        if self.umin is not None:
            self.constraints.append(self.filtered_control >= self.umin)
        if self.umax is not None:
            self.constraints.append(self.filtered_control <= self.umax)
        self.QP = cp.Problem(self.obj, self.constraints)
        assert self.QP.is_qp(), "This is not a quadratic program"

    def set_constraint(self, Lf_h: Array, Lg_h: Array, h: float):
        self.b.value = np.atleast_1d(self.alpha(h) + Lf_h)
        self.A.value = np.atleast_2d(Lg_h)

    def __call__(self, state: Array, time: float = 0.0, nominal_control=None) -> Array:
        if not hasattr(self, "QP"):
            self.setup_optimization_problem()
        self.set_nominal_control(state, time, nominal_control)
        return self.u(state, time)

    def u(self, state: Array, time: float = 0.0):
        h = np.atleast_1d(self.cbf.vf(state, time))
        Lf_h, Lg_h = self.cbf.lie_derivatives(state, time)
        opt_sols = []
        if state.ndim == 1:
            state = state[None, ...]
        for i in range(state.shape[0]):
            self.set_constraint(Lf_h[i], Lg_h[i], h[i])
            self.nominal_control_cp.value = np.atleast_1d(self.nominal_control[i])
            self._solve_problem()
            opt_sols.append(np.atleast_1d(self.opt_sol))
        opt_sols = np.array(opt_sols)
        # logger.warning("A, b, nominal_control: ", self.alpha(h) + Lf_h, Lg_h, self.nominal_control, opt_sols)
        # raise ValueError("STOP HERE")

        return opt_sols

    def _solve_problem(self):
        """Lower level function to solve the optimization problem"""
        solver_failure = False
        try:
            self.QP.solve(solver=self.solver, verbose=self.verbose)
            self.opt_sol = self.filtered_control.value
        except (cp.SolverError, ValueError):
            solver_failure = True
        if self.QP.status in ["infeasible", "unbounded"] or solver_failure:
            # logger.warning("QP solver failed")
            if (self.umin is None) and (self.umax is None):
                # logger.warning("Returning nominal control value")
                self.opt_sol = self.nominal_control_cp.value
            else:
                if self.umin is not None and self.umax is not None:
                    # TODO: This should depend on "controlMode"
                    # logger.warning("Returning safest possible control")
                    self.opt_sol = (
                        np.int64(self.A.value >= 0) * self.umax + np.int64(self.A.value < 0) * self.umin
                    ).reshape(-1)
                elif (self.A.value >= 0).all() and self.umax is not None:
                    # logger.warning("Returning umax")
                    self.opt_sol = self.umax
                elif (self.A.value <= 0).all() and self.umin is not None:
                    # logger.warning("Returning umin")
                    self.opt_sol = self.umin
                else:
                    # logger.warning("Returning nominal control value")
                    self.opt_sol = self.nominal_control_cp.value
                # elif self.umax is not None:
                #     self.opt_sol = (
                #         np.int64(self.A.value >= 0) * self.umax
                #         + np.int64(self.A.value < 0) * self.nominal_control_cp.value
                #     ).reshape(-1)
                # elif self.umin is not None:
                #     self.opt_sol = (
                #         np.int64(self.A.value >= 0) * self.nominal_control_cp.value
                #         + np.int64(self.A.value < 0) * self.umin
                #     ).reshape(-1)

# used to test solving time through casadi, which is slower than cvxpy
                    
class ControlAffineASIF_casadi(ASIF):
    def __init__(self, dynamics: ControlAffineDynamics, cbf: ControlAffineCBF, test: bool = True, **kwargs) -> None:
        super().__init__(dynamics, cbf, test, **kwargs)

        self.opti = ca.Opti()
        self.filtered_control = self.opti.variable(self.dynamics.control_dims)

        self.nominal_control_cp = self.opti.parameter(self.dynamics.control_dims)
        self.b = self.opti.parameter(1,)
        self.A = self.opti.parameter(1, self.dynamics.control_dims)

        self.umin = kwargs.get("umin")
        self.umax = kwargs.get("umax")

        self.opt_sol = np.zeros(self.filtered_control.shape)

        if test:
            test_asif.test_control_affine_asif(self)

    def setup_optimization_problem(self):
        """
        min || u - u_des ||^2
        s.t. A @ u + b >= 0
        """
        self.constraints = [self.A @ self.filtered_control + self.b >= 0]
        if self.umin is not None:
            self.constraints.append(self.filtered_control >= self.umin)
        if self.umax is not None:
            self.constraints.append(self.filtered_control <= self.umax)

        self.opti.subject_to(self.constraints)
        self.obj = (self.filtered_control - self.nominal_control_cp) ** 2
        self.opti.minimize(self.obj)

    def __call__(self, state: Array, time: float = 0.0, nominal_control=None) -> Array:
        if not hasattr(self, "obj"):
            self.setup_optimization_problem()
        self.set_nominal_control(state, time, nominal_control)
        return self.u(state, time)

    def u(self, state: Array, time: float = 0.0):
        opt_sols = []
        self.h = np.atleast_1d(self.cbf.vf(state, time))
        self.Lf_h, self.Lg_h = self.cbf.lie_derivatives(state, time)
        self.b_value, self.A_value = np.atleast_1d(self.alpha(self.h) + self.Lf_h), np.atleast_2d(self.Lg_h)
        if state.ndim == 1:
            state = state[None, ...]
        for i in range(state.shape[0]):
            self.opti.set_value(self.b, self.b_value)
            self.opti.set_value(self.A, self.A_value)
            self.opti.set_value(self.nominal_control_cp, np.atleast_1d(self.nominal_control[i]))
            self._solve_problem()
            opt_sols.append(np.atleast_1d(self.opt_sol))
        opt_sols = np.array(opt_sols)
        # logger.warning("nominal control, opt control, constraints", self.nominal_control, opt_sols, self.constraints)
        # raise ValueError("STOP HERE")

        return opt_sols

    def _solve_problem(self):
        """Lower level function to solve the optimization problem"""
        try:
            p_opts = {"print_time": False, "verbose": False}
            s_opts = {"print_level": 0}
            self.opti.solver("ipopt", p_opts, s_opts)
            sol = self.opti.solve()
            self.opt_sol = sol.value(self.filtered_control)
        except:
            # choose the safest action
            self.opt_sol = (
                        np.int64(self.A_value >= 0) * self.umax + np.int64(self.A_value < 0) * self.umin
                    ).reshape(-1)

class ImplicitASIF(metaclass=abc.ABCMeta):
    def __init__(self, dynamics: Dynamics, cbf: ImplicitCBF, backup_controller: BackupController, **kwargs) -> None:
        self.dynamics = dynamics
        self.cbf = cbf
        self.nominal_control = None
        self.backup_controller = backup_controller
        self.verify_every_x = kwargs.get("verify_every_x", 1)
        self.n_backup_steps = int(self.backup_controller.T_backup / self.dynamics.dt)
        self.alpha_backup = kwargs.get("alpha_backup", lambda x: x)
        self.alpha_safety = kwargs.get("alpha_safety", lambda x: x)
        self.nominal_policy = kwargs.get("nominal_policy", lambda x, t: 0)

    @abc.abstractmethod
    def __call__(self, state: Array, nominal_control=None, time: float = 0.0) -> Array:
        """Implements the active safety invariance filter"""