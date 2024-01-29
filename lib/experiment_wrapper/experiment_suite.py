"""A collection suite manages a collection of experiments, allowing the user to run all of them in a single run."""

from experiment_wrapper import Experiment
from typing import List, Tuple
import pandas as pd
import os
from matplotlib.figure import Figure


class ExperimentSuite:
    def __init__(self, experiments: List[Experiment]):
        self.experiments = experiments

    def run_all(self, dynamics, controllers_under_test) -> List[pd.DataFrame]:
        results = []
        for i, experiment in enumerate(self.experiments):
            results.append(experiment.run(dynamics[i], controllers_under_test[i]))
        return results

    def run_all_and_save_to_csv(self, dynamics, controllers_under_test, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        subdir = f"{save_dir}/{timestamp}"  # Save all experiments to subdirectory
        for i, experiment in enumerate(self.experiments):
            experiment.run_and_save_to_csv(dynamics[i], controllers_under_test[i], subdir)

    def run_all_and_plot(
        self, dynamics, controllers_under_test, display_plots: bool = False
    ) -> List[Tuple[str, Figure]]:
        fig_handles = []
        for i, experiment in enumerate(self.experiments):
            fig_handles += experiment.run_and_plot(
                dynamics[i], controllers_under_test[i], display_plots
            )
        return fig_handles
