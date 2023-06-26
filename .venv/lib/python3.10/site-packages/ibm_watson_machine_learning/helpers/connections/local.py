#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


import sys
import pandas as pd


class LocalBatchReader:
    """LocalBatchReader is designed to """
    def __init__(self, file_path: str, batch_size: int = 1073741824 // 10):
        self.file_path = file_path
        self.batch_size = batch_size    # default 100 MB
        self.row_size = 0

        self._determine_row_size()

    def _determine_row_size(self) -> None:
        data_row = pd.read_csv(self.file_path, chunksize=1)
        self.row_size = sys.getsizeof(data_row)

    def _calculate_chunk_size(self) -> int:
        return self.batch_size // self.row_size

    def __iter__(self) -> pd.DataFrame:
        x = self._calculate_chunk_size()
        return pd.read_csv(self.file_path, chunksize=self._calculate_chunk_size())
