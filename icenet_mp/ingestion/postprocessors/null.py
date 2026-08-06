from pathlib import Path

from .ipostprocessor import IPostprocessor


class NullPostprocessor(IPostprocessor):
    def process(
        self,
        path_dataset: Path,  # noqa: ARG002
        *,
        overwrite: bool,  # noqa: ARG002
    ) -> None:
        """NullPostprocessor does not generate any postprocessing artifacts."""
        return
