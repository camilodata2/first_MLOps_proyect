#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

__all__ = ["ExperimentDataLoader"]

import pandas as pd

# Note: Use torch DataLoader and Dataset if available, if not, use mocked ones.
# SimpleDataLoader is the simplest wrapper ever for such thing, it is not compatible with torch anymore.
try:
    from torch.utils.data import DataLoader, IterableDataset

except ImportError:

    class SimpleDataLoader:
        def __init__(self, dataset, collate_fn):
            self.dataset = dataset
            self.collate_fn = collate_fn

        def __iter__(self):
            return (data for data in self.dataset)

    DataLoader = SimpleDataLoader
    IterableDataset = object
# --- end note


class ExperimentDataLoader(DataLoader):
    """Experiment Data Loader based on torch DataLoader.
    It interacts with the Flight Service to provide data batch stream

    **Example**

    .. code-block:: python

        experiment_metadata = {
            "prediction_column": 'species',
            "prediction_type": "classification",
            "project_id": os.environ.get('PROJECT_ID'),
            'wml_credentials': wml_credentials
        }

        connection = DataConnection(data_asset_id='5d99c11a-2060-4ef6-83d5-dc593c6455e2')


        iterable_dataset = ExperimentIterableDataset(connection=connection,
                                                     enable_sampling=False,
                                                     experiment_metadata=experiment_metadata)

        data_loader = ExperimentDataLoader(dataset=iterable_dataset)

        for data in data_loader:
            print(data)
    """

    def __init__(self, dataset: "IterableDataset"):
        super().__init__(dataset=dataset, collate_fn=custom_collate)


def custom_collate(batch: pd.DataFrame) -> pd.DataFrame:
    """Custom collate function for DataLoader, it simply get and return unchanged pandas DF"""
    if isinstance(batch, list):
        return batch[0]

    else:
        return batch
