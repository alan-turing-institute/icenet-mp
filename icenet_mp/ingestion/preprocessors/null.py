from pathlib import Path

from .ipreprocessor import IPreprocessor


class NullPreprocessor(IPreprocessor):
    def download(self, preprocessor_path: Path) -> None:  # noqa: ARG002
        """NullPreprocessor does not download any data."""
        return
