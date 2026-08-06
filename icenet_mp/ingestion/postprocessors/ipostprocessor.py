from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any


class IPostprocessor(ABC):
    def __init__(self, config: MutableMapping[str, Any]) -> None:
        """Initialise the IPostprocessor base class."""
        self.base_path = Path(config["base_path"]).resolve()
        self.cls_name = str(config.get("postprocessor", {}).get("type", "None"))
        self.dataset_name = str(config.get("name", "None"))

    @abstractmethod
    def process(self, path_dataset: Path, *, overwrite: bool) -> None:
        """Generate postprocessing artifacts for the finalised dataset."""
