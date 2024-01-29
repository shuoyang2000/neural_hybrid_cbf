from copy import copy
import tqdm
from experiment_wrapper import Experiment, ScenarioList, Controllers, Dynamics
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import warnings
import logging

logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=FutureWarning)


class RolloutTrajectory(Experiment):
    def __init__(
        self,
        name: str,
        start_x: np.ndarray,
        x_indices: Optional[List[int]] = None,
        x_labels: Optional[List[str]] = None,
        u_indices: Optional[List[int]] = None,
        u_labels: Optional[List[str]] = None,
        scenarios: Optional[ScenarioList] = None,
        n_sims_per_start: int = 5,
        t_sim: float = 5.0,
    ):
        """Initialize the rollout trajectory experiment. Optionally run multiple scenarios (e.g. different stochastic
        policies / disturbances / uncertain parameters) from multiple starting states

        Args:
            name (str): Name of the experiment
            start_x (np.ndarray): Starting states [n_start_states, n_dims]
            x_indices (List[int]): A list of the indices of state variables to log
            u_indices (List[int]): A list of the indices of control variables to log
            scenarios (Optional[ScenarioList], optional): Different scenarios. Defaults to None.
            n_sims_per_start (int, optional): . Defaults to 5.
            t_sim (float, optional): _description_. Defaults to 5.0.
        """
        super().__init__(name)
        self.start_x = np.atleast_2d(start_x)
        self.x_indices = x_indices
        self.x_labels = x_labels
        self.u_indices = u_indices
        self.u_labels = u_labels
        self.scenarios = scenarios
        self.n_sims_per_start = n_sims_per_start  # For random disturbances
        self.t_sim = t_sim

    def set_idx_and_labels(self, dynamics: Dynamics):
        if self.x_indices is None:  # FIXME: None or [] for initialization?
            self.x_indices = list(range(dynamics.n_dims))
        # Default to saving all controls
        if self.u_indices is None:
            self.u_indices = list(range(dynamics.control_dims))
        if self.x_labels is None:
            self.x_labels = [dynamics.STATES[idi] for idi in self.x_indices]
        if self.u_labels is None:
            self.u_labels = [dynamics.CONTROLS[idi] for idi in self.u_indices]


    def run(self, dynamics: Dynamics, controllers: Controllers, control_bounds: np.ndarray = None) -> pd.DataFrame:
        """Overrides Experiment.run for rollout trajectory experiments. Same args as Experiment.run

        At every time step:
        1) Check whether the control needs to be updated
        2) Log the current data
        3) Take step in the simulation
        """
        if not isinstance(controllers, dict):
            controllers = {controllers.__class__.__name__: controllers}
        self.set_idx_and_labels(dynamics)

        results = []
        n_sims = self.n_sims_per_start * self.start_x.shape[0]
        x_sim_start = np.zeros((n_sims, dynamics.n_dims))

        for controller_name, controller in controllers.items():
            for i in range(0, self.start_x.shape[0]):
                for j in range(0, self.n_sims_per_start):
                    x_sim_start[i * self.n_sims_per_start + j, :] = self.start_x[i, :]

            x_current = x_sim_start
            u_current = np.zeros((n_sims, dynamics.control_dims))
            if hasattr(controller, "reset_controller"):
                controller.reset(x_current)

            delta_t = dynamics.dt
            controller_update_freq = (
                int(controller.controller_dt / delta_t) if hasattr(controller, "controller_dt") else 1
            )
            num_steps = int(self.t_sim / delta_t)

            prog_bar_range = tqdm.tqdm(range(0, num_steps), desc="Controller rollout")

            for tstep in prog_bar_range:
                t = tstep * delta_t

                ######## UPDATE CONTROLLER ########
                if tstep % controller_update_freq == 0:

                    u_current = controller(x_current, t).reshape(n_sims, dynamics.control_dims)
                    if control_bounds is not None:
                        u_current = np.clip(u_current, control_bounds[0], control_bounds[1])
                ########### LOGGING ###############
                for sim_index in range(n_sims):
                    base_log_packet = {"t": t}
                    base_log_packet["controller"] = controller_name
                    base_log_packet["scenario"] = sim_index % self.n_sims_per_start
                    base_log_packet["rollout"] = sim_index // self.n_sims_per_start

                    if hasattr(controller, "save_info"):
                        base_log_packet.update(controller.save_info(x_current[sim_index], u_current[sim_index], t))

                    for i, state_index in enumerate(self.x_indices):
                        log_packet = copy(base_log_packet)
                        log_packet["measurement"] = self.x_labels[i]
                        log_packet["value"] = x_current[sim_index, state_index]
                        results.append(log_packet)

                    for i, control_index in enumerate(self.u_indices):
                        log_packet = copy(base_log_packet)
                        log_packet["measurement"] = self.u_labels[i]
                        log_packet["value"] = u_current[sim_index, control_index]
                        results.append(log_packet)

                    if hasattr(controller, "save_measurements"):
                        for key, value in controller.save_measurements(
                            x_current[sim_index], u_current[sim_index], t
                        ).items():
                            log_packet = copy(base_log_packet)
                            log_packet["measurement"] = key
                            log_packet["value"] = value
                            results.append(log_packet)

                ########### SIMULATION ###############
                x_current = dynamics.step(x_current, u_current, t)
            # if self.save_location is not None:
            #     print("Saved results to " + self.save_location + ".csv")
            #     pd.DataFrame(results).to_csv(self.save_location + ".csv")
        return pd.DataFrame(results)

    def run_hybrid(self, dynamics1: Dynamics, dynamics2: Dynamics, controllers1: Controllers, controllers2: Controllers, switch) -> pd.DataFrame:
        """Overrides Experiment.run for rollout trajectory experiments. Same args as Experiment.run

        At every time step:
        1) Check whether the control needs to be updated
        2) Log the current data
        3) Take step in the simulation
        """
        if not isinstance(controllers1, dict):
            controllers = {controllers1.__class__.__name__: controllers1}
        dynamics = dynamics1
        self.set_idx_and_labels(dynamics)

        results = []
        n_sims = self.n_sims_per_start * self.start_x.shape[0]
        x_sim_start = np.zeros((n_sims, dynamics.n_dims))
        
        

        for controller_name, controller in controllers1.items():
            jump_index = 0 # 0 is the first mode, 1 is the second mode
            for i in range(0, self.start_x.shape[0]):
                for j in range(0, self.n_sims_per_start):
                    x_sim_start[i * self.n_sims_per_start + j, :] = self.start_x[i, :]

            x_current = x_sim_start
            print("x_current (position, velocity, distance): ", x_current, x_current.shape)
            u_current = np.zeros((n_sims, dynamics.control_dims))
            traj_all = x_current
            control_all = u_current
            if hasattr(controller, "reset_controller"):
                controller.reset(x_current)

            delta_t = dynamics.dt
            controller_update_freq = (
                int(controller.controller_dt / delta_t)
                if hasattr(controller, "controller_dt")
                else 1
            )
            num_steps = int(self.t_sim / delta_t)

            prog_bar_range = tqdm.tqdm(range(0, num_steps), desc="Controller rollout")

            for tstep in prog_bar_range:
                t = tstep * delta_t

                ######## UPDATE CONTROLLER ########
                if tstep % controller_update_freq == 0:
                    u_current = controller(x_current, t).reshape(n_sims, dynamics.control_dims)

                ########### LOGGING ###############
                for sim_index in range(n_sims):
                    base_log_packet = {"t": t}
                    base_log_packet["controller"] = controller_name
                    base_log_packet["scenario"] = sim_index % self.n_sims_per_start
                    base_log_packet["rollout"] = sim_index // self.n_sims_per_start

                    if hasattr(controller, "save_info"):
                        base_log_packet.update(
                            controller.save_info(x_current[sim_index], u_current[sim_index], t)
                        )

                    for i, state_index in enumerate(self.x_indices):
                        log_packet = copy(base_log_packet)
                        log_packet["measurement"] = self.x_labels[i]
                        log_packet["value"] = x_current[sim_index, state_index]
                        #print("log value: ", log_packet["value"])
                        results.append(log_packet)

                    for i, control_index in enumerate(self.u_indices):
                        log_packet = copy(base_log_packet)
                        log_packet["measurement"] = self.u_labels[i]
                        log_packet["value"] = u_current[sim_index, control_index]
                        results.append(log_packet)

                    if hasattr(controller, "save_measurements"):
                        for key, value in controller.save_measurements(
                            x_current[sim_index], u_current[sim_index], t
                        ).items():
                            log_packet = copy(base_log_packet)
                            log_packet["measurement"] = key
                            log_packet["value"] = value
                            results.append(log_packet)

                ########### SIMULATION ###############
                x_current = dynamics.step(x_current, u_current, t)
                traj_all = np.concatenate((traj_all, x_current), axis=0)
                control_all = np.concatenate((control_all, u_current), axis=0)

                if x_current[0, 0] >= switch and jump_index == 0:
                    if jump_index == 0:
                        print("switching time: ", t)
                    controller = controllers2[controller_name]
                    jump_index = 1
                    jump_state = x_current
                    dynamics = dynamics2

        return pd.DataFrame(results), jump_state, traj_all, control_all


class TimeSeriesExperiment(RolloutTrajectory):
    def plot(
        self,
        dynamics: Dynamics,
        results_df: pd.DataFrame,
        extra_measurements: list = [],
        display_plots: bool = False,
        **kwargs
    ) -> List[Tuple[str, Figure]]:
        """Overrides Experiment.plot to plot the time series of the measurements. Same args as Experiment.plot, but also:

        Extra Args:
            extra_measurements (list, optional): other variables (beyond x_labels and y_labels to display).
        """
        self.set_idx_and_labels(dynamics)
        sns.set_theme(context="talk", style="white")
        default_colors = sns.color_palette("colorblind")
        colors = kwargs.get("colors", default_colors)
        alpha = kwargs.get("alpha", [1] * len(results_df.controller.unique()))
        linestyles = kwargs.get("linestyles", ["-"] * len(results_df.controller.unique()))
        extra_measurements = copy(extra_measurements)
        for measurement in extra_measurements:
            if measurement not in results_df.measurement.values:
                logger.warning("Measurement {} not in results dataframe".format(measurement))
                extra_measurements.remove(measurement)
        axs = kwargs.get("axs")
        num_plots = len(self.x_indices) + len(self.u_indices) + len(extra_measurements)

        if axs is None:
            fig, axs = plt.subplots(num_plots, 1, sharex=True)
            fig.set_size_inches(10, 4 * num_plots)
        else:
            assert axs.shape[0] == num_plots
            fig = axs[0].get_figure()

        axs = np.array(axs)  # Also a np.array for num_plots = 1

        num = -1
        for controller in results_df.controller.unique():
            num += 1
            for scenario in results_df.scenario.unique():
                for rollout in results_df.rollout.unique():
                    mask = (
                        (results_df.controller == controller)
                        & (results_df.scenario == scenario)
                        & (results_df.rollout == rollout)
                    )

                    for i, state_label in enumerate(self.x_labels):
                        ax = axs[i]
                        state_mask = mask & (results_df.measurement.values == state_label)
                        ax.plot(
                            results_df[state_mask].t,
                            results_df[state_mask].value,
                            color=colors[num],
                            alpha=alpha[num],
                            ls=linestyles[num],
                        )
                        ax.set_ylabel(state_label)

                    for i, control_label in enumerate(self.u_labels):
                        ax = axs[len(self.x_labels) + i]
                        control_mask = mask & (results_df.measurement.values == control_label)
                        ax.plot(
                            results_df[control_mask].t,
                            results_df[control_mask].value,
                            color=colors[num],
                            alpha=alpha[num],
                            ls=linestyles[num],
                        )
                        ax.set_ylabel(control_label)
                        ax.set_ylabel(control_label)

                    for i, extra_label in enumerate(extra_measurements):
                        ax = axs[len(self.x_labels) + len(self.u_labels) + i]
                        extra_mask = mask & (results_df.measurement.values == extra_label)
                        ax.plot(
                            results_df[extra_mask].t,
                            results_df[extra_mask].value,
                            color=colors[num],
                            alpha=alpha[num],
                            ls=linestyles[num],
                        )
                        ax.set_ylabel(extra_label)

        axs[-1].set_xlabel("t")
        axs[-1].set_xlim(min(results_df.t), max(results_df.t))

        fig_handle = ("Rollout (time series)", fig)

        if display_plots:
            plt.show()

        return [fig_handle]


class StateSpaceExperiment(RolloutTrajectory):
    def plot(
        self, dynamics: Dynamics, results_df: pd.DataFrame, display_plots: bool = False, **kwargs
    ) -> List[Tuple[str, Figure]]:
        """Overrides Experiment.plot to plot state space data. Same args as Experiment.plot"""
        self.set_idx_and_labels(dynamics)
        assert len(self.x_labels) in [2, 3], "Can't plot in this dimension!"

        ax = kwargs.get("ax")
        alpha = kwargs.get("alpha", [1] * len(results_df.controller.unique()))
        linestyles = kwargs.get("linestyles", ["-"] * len(results_df.controller.unique()))
        # 2D visualization
        if len(self.x_labels) == 2:

            if ax is None:
                fig, ax = plt.subplots()
                fig.set_size_inches(9, 6)
            else:
                fig = ax.get_figure()
            i = -1
            for controller in results_df.controller.unique():
                i += 1
                for scenario in results_df.scenario.unique():
                    for rollout in results_df.rollout.unique():
                        mask = (
                            (results_df.controller == controller)
                            & (results_df.scenario == scenario)
                            & (results_df.rollout == rollout)
                        )
                        xmask = mask & (results_df.measurement.values == self.x_labels[0])
                        ymask = mask & (results_df.measurement.values == self.x_labels[1])
                        xvals = results_df[xmask].value.values
                        yvals = results_df[ymask].value.values

                        if kwargs.get("color") is None:
                            l = ax.plot(xvals, yvals, alpha=alpha[i])
                        else:
                            l = ax.plot(
                                xvals,
                                yvals,
                                color=kwargs.get("color")[i],
                                alpha=alpha[i],
                                ls=linestyles[i],
                            )
                        if kwargs.get("add_direction", True):
                            add_arrow(l[0], direction="right", position=(xvals[0] + xvals[-1]) / 2)
                        ax.plot(xvals[0], yvals[0], "o", color=l[0].get_color(), alpha=alpha[i])
                        ax.plot(xvals[-1], yvals[-1], "x", color=l[0].get_color(), alpha=alpha[i])
            ax.set_xlabel(self.x_labels[0])
            ax.set_ylabel(self.x_labels[1])
        # 3D visualization
        else:
            raise NotImplementedError("Future work!")

        fig_handle = ("State space visualization", fig)
        return [fig_handle]


def add_arrow(line, position=None, direction="right", size=25, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position - 5 + 10 * np.random.rand()))
    if direction == "right":
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate(
        "",
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="simple", color=color),
        size=size,
    )