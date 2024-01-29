from typing import Any, Dict, List


class Dynamics:
    """Provides a template for the functionality required from a dynamics class to interface with the experiment
    wrapper functionality.

    A dynamics class must implement the following methods:
    - n_dims: returns the number of dimensions of the state space
    - control_dims: returns the number of dimensions of the control space
    - dt: returns the time step of the dynamics
    - step: takes in the current state, control and time and returns the next state"""

    STATES: List[str]
    CONTROLS: List[str]

    def __init__(self):
        self._n_dims: int
        self._control_dims: int
        self._dt: float
        raise RuntimeError("Dynamics is a template class")

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def control_dims(self) -> int:
        return self._control_dims

    @property
    def dt(self) -> float:
        return self._dt

    def step(self, x: Any, u: Any, t: float) -> Any:
        pass


class Controller:
    """Provides a template for the functionality required from a controller class to interface with the experiment
    wrappper functionality.

    A controller class must implement the following methods:
    - __call__: takes in the current state and time and returns the control (note: a function object can be used, e.g.:
    def nominal_policy(x, t):
        return L @ x
    with L the LQR controller matrix"""

    def __init__(self):
        raise RuntimeError("Controller is a template class")

    def __call__(self, x: Any, t: float) -> Any:
        pass


class ExtendedController(Controller):
    """Provides a template for functionality that is optional called within the experiment wrapper functionality.

    A controller class (in addition to being callable) can also implement the following methods:
    - controller_dt: returns the time step of the controller
    - save_info: takes in the current state, control and time and returns a dictionary of information to be saved for
    all measurements
    - save_measurements: takes in the current state, control and time and returns a dictionary of additional
    measurements to be saved
    - reset: takes in the current state and resets the controller to an initial state
    """

    def __init__(self):
        self._controller_dt: float
        raise RuntimeError("ExtendedController is a template class")

    @property
    def controller_dt(self) -> float:
        return self._controller_dt

    def save_info(self, x: Any, u: Any, t: float) -> Dict[str, Any]:
        return {}

    def save_measurements(self, x: Any, u: Any, t: float) -> Dict[str, Any]:
        return {}

    def reset(self, x: Any) -> None:
        pass


from experiment_wrapper.experiment import Experiment, ScenarioList, Controllers
from experiment_wrapper.rollout_trajectory import (
    RolloutTrajectory,
    TimeSeriesExperiment,
    StateSpaceExperiment,
)
from experiment_wrapper.experiment_suite import ExperimentSuite

__version__ = "1.1.0"
