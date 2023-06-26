#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from .progress_bar import ProgressBar
from .utils import is_ipython

__all__ = [
    "LocalTrainingMessageHandler"
]


class LocalTrainingMessageHandler:
    """Show progress bar during local AutoPipelines training/fit."""
    def __init__(self):
        self.progress_bar = None
        self.progress_bar_1 = None
        self.previous_stage = None

        # note: only notebook version
        if is_ipython() and False:
            self.total = 100
            self.progress_bar_1 = ProgressBar(desc="Total", total=self.total, position=0, ncols='100%')
            self.progress_bar_2 = ProgressBar(desc="Waiting", total=self.total, leave=False, ncols='100%')

        # note: only console version
        else:
            self.total = 200
            self.progress_bar = ProgressBar(desc="Total", total=self.total, position=0, ncols=100)

    def on_training_message(self, status_state):
        """This method should be used by the ai4ml estimator/optimization process during training."""

        if 'ml_metrics' in status_state:
            metric = status_state['ml_metrics']
            for stage in ['pre_hpo_d_output', 'hpo_d_output', 'hpo_c_output', 'fold_output',
                          'cognito_output', 'compose_model_type_output', 'global_output', 'daub_running_output',
                          'cognito_running_output', 'hpo_c_running_output', 'hpo_d_running_output']:

                if (stage in metric) and (metric[stage] is not None):

                    if self.progress_bar is not None:
                        self.progress_bar.set_description(desc=stage)
                        if self.total - self.progress_bar.counter <= 5:
                            pass

                        else:
                            self.progress_bar.increment_counter(progress=1)
                            self.progress_bar.update()

                    else:

                        if self.previous_stage != stage:
                            self.previous_stage = stage
                            self.progress_bar_2.last_update()
                            self.progress_bar_2.close()
                            self.progress_bar_2 = ProgressBar(desc="Waiting", total=self.total, leave=False, ncols='100%')
                            if self.total - self.progress_bar_1.counter <= 10:
                                pass

                            else:
                                self.progress_bar_1.increment_counter(progress=5)
                                self.progress_bar_1.update()

                        self.progress_bar_2.set_description(desc=stage)
                        if self.total - self.progress_bar_1.counter <= 1:
                            pass

                        else:
                            self.progress_bar_2.increment_counter(progress=1)
                            self.progress_bar_2.update()
