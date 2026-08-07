from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig

if TYPE_CHECKING:
    from .ipreprocessor import IPreprocessor


class CompositePreprocessor:
    def __init__(
        self, dataset_name: str, dataset_config: DictConfig, base_path: Path
    ) -> None:
        """Initialise a CompositePreprocessor from its child preprocessors."""
        self.children: list[IPreprocessor] = [
            hydra.utils.instantiate(
                spec, base_path=base_path, dataset_name=dataset_name
            )
            for spec in dataset_config.get("preprocessors", {}).values()
        ]

    def process(self, *, overwrite: bool) -> None:
        """Process the dataset with each child preprocessor, in order."""
        for child in self.children:
            child.process(overwrite=overwrite)
