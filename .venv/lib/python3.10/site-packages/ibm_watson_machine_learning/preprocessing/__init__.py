#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ..utils.autoai.utils import try_import_graphviz

try_import_graphviz()

from .multiple_files_preprocessor import DataJoinGraph
from .data_join_pipeline import DataJoinPipeline
