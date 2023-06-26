#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Union
import sys

from ibm_watson_machine_learning.utils.autoai.utils import try_import_tqdm

try_import_tqdm()

from tqdm import tqdm as TQDM

__all__ = [
    "ProgressBar"
]


class ProgressBar(TQDM):
    """Progress Bar class for handling progress bar display. It is based on 'tqdm' class, could be extended."""

    def __init__(self, ncols: Union[str, int], position: int = 0, desc: str = None, total: int = 100,
                 leave: bool = True, bar_format: str = '{desc}: {percentage:3.0f}%|{bar}|') -> None:
        """
        :param desc: description string to be added as a prefix to progress bar
        :type desc: str, optional

        :param total: total length of the progress bar
        :type total: int, optional
        """
        # note: to see possible progress bar formats please look at super 'bar_format' description
        super().__init__(desc=desc, total=total, leave=leave, position=position, ncols=ncols, file=sys.stdout,
                         bar_format=bar_format)
        self.total = total
        self.previous_message = None
        self.counter = 0
        self.progress = 0

    def increment_counter(self, progress: int = 5) -> None:
        """Increment internal counter and waits for specified time.

        :param progress: number of steps at a time progress bar
        :type progress: int, optional
        """
        self.progress = progress
        self.counter += progress

    def reset_counter(self) -> None:
        """Restart internal counter."""
        self.counter = 0

    def update(self):
        """Update the counter with specific progress."""
        super().update(n=self.progress)

    def last_update(self):
        """Fill up the progress bar till the end, this was the last run."""
        super().update(n=self.total - self.counter)
