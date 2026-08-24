from pathlib import Path

import numpy as np

from icenet_mp.data.single_dataset import SingleDataset


def test_date_lookups_normalise_to_noon(
    mock_dataset_non_normalized_times: Path,
) -> None:
    """Confirm that a midnight datetime is normalised during lookup."""
    dataset = SingleDataset(
        name="test_normalized",
        input_files=[mock_dataset_non_normalized_times],
    )
    midnight = np.datetime64("2020-01-01")

    assert dataset.to_index(midnight) == 0
    np.testing.assert_array_equal(
        dataset.get_tchw_slice(midnight, 1),
        dataset.get_tchw([dataset.dates[0]]),
    )
