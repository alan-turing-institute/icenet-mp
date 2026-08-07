from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig

if TYPE_CHECKING:
    from .ipostprocessor import IPostprocessor


class CompositePostprocessor:
    def __init__(
        self, dataset_name: str, dataset_config: DictConfig, base_path: Path
    ) -> None:
        """Initialise a CompositePostprocessor from its child postprocessors."""
        self.children: list[IPostprocessor] = [
            hydra.utils.instantiate(
                spec, base_path=base_path, dataset_name=dataset_name
            )
            for spec in dataset_config.get("postprocessors", {}).values()
        ]

    def process(self, path_dataset: Path, *, overwrite: bool) -> None:
        """Process the dataset with each child postprocessor, in order."""
        for child in self.children:
            child.process(path_dataset, overwrite=overwrite)
