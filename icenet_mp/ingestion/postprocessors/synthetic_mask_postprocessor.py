import logging
from pathlib import Path

import numpy as np
from anemoi.datasets import open_dataset

from .ipostprocessor import IPostprocessor

logger = logging.getLogger(__name__)


class SyntheticMaskPostprocessor(IPostprocessor):
    def process(self, path_dataset: Path, path_masks: Path, *, overwrite: bool) -> None:
        """Generate all-active land and active grid cell masks for a synthetic dataset."""
        path_masks.mkdir(parents=True, exist_ok=True)
        land_mask_path = path_masks / "land_mask.npy"
        active_mask_path = path_masks / "active_mask.npy"
        if land_mask_path.exists() and active_mask_path.exists() and not overwrite:
            logger.debug("Both masks already exist, skipping creation.")
            return

        field_shape = open_dataset(path_dataset).field_shape[-2:]
        synthetic_mask = np.ones(field_shape, dtype=np.uint8)
        np.save(land_mask_path, synthetic_mask)
        np.save(active_mask_path, synthetic_mask)
        logger.info(
            "Created all-active masks for synthetic dataset %s.", self.dataset_name
        )
