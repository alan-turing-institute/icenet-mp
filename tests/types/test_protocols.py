from omegaconf import DictConfig

from icenet_mp.types.protocols import SupportsMetadata


class _WithMetadata:
    def set_metadata(self, config: DictConfig, model_name: str) -> None:
        self.config = config
        self.model_name = model_name


class _WithoutMetadata:
    pass


class TestSupportsMetadata:
    """Tests for the SupportsMetadata protocol."""

    def test_isinstance_false_when_set_metadata_is_missing(self) -> None:
        """Reject an object that has no set_metadata method."""
        assert not isinstance(_WithoutMetadata(), SupportsMetadata)

    def test_isinstance_true_for_matching_method_signature(self) -> None:
        """Accept an object that duck-types a matching set_metadata method."""
        assert isinstance(_WithMetadata(), SupportsMetadata)
