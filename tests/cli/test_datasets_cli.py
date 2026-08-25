import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from icenet_mp.cli.main import app


class TestCreateCLI:
    class FakeDownloader:
        """A minimal downloader stub exposing only what `create` needs."""

        def __init__(self, name: str, *, error: Exception | None = None) -> None:
            """Store the downloader's name and an optional error to raise."""
            self.name = name
            self.error = error
            self.create_calls: list[bool] = []

        def create(self, *, overwrite: bool) -> None:
            """Record the overwrite flag and raise the configured error, if any."""
            self.create_calls.append(overwrite)
            if self.error is not None:
                raise self.error

    def test_calls_create_on_each_downloader_with_overwrite_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Call create(overwrite=True) on every downloader when --overwrite is set."""
        first = self.FakeDownloader("first")
        second = self.FakeDownloader("second")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [first, second]
        )

        result = CliRunner().invoke(
            app, ["datasets", "create", "--overwrite"], prog_name="imp"
        )

        assert result.exit_code == 0, result.output
        assert first.create_calls == [True]
        assert second.create_calls == [True]

    def test_defaults_overwrite_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default --overwrite to False when not passed."""
        downloader = self.FakeDownloader("example")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [downloader]
        )

        result = CliRunner().invoke(app, ["datasets", "create"], prog_name="imp")

        assert result.exit_code == 0, result.output
        assert downloader.create_calls == [False]

    def test_exits_with_error_and_stops_on_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exit 1 and skip remaining downloaders when one fails to create."""
        failing = self.FakeDownloader("failing", error=RuntimeError("boom"))
        never_reached = self.FakeDownloader("never-reached")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders",
            lambda _config: [failing, never_reached],
        )

        with caplog.at_level(logging.ERROR):
            result = CliRunner().invoke(app, ["datasets", "create"], prog_name="imp")

        assert result.exit_code == 1
        assert "Failed to create failing: boom" in caplog.text
        assert never_reached.create_calls == []


class TestInspectCLI:
    class FakeDownloader:
        """A minimal downloader stub exposing only what `inspect` needs."""

        def __init__(self, name: str, *, error: Exception | None = None) -> None:
            """Store the downloader's name and an optional error to raise."""
            self.name = name
            self.error = error
            self.inspect_calls: list[bool] = []

        def inspect(self, *, verbose: bool) -> None:
            """Record the verbose flag and raise the configured error, if any."""
            self.inspect_calls.append(verbose)
            if self.error is not None:
                raise self.error

    def test_calls_inspect_on_each_downloader_with_verbose_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Call inspect(verbose=True) on every downloader when --verbose is set."""
        first = self.FakeDownloader("first")
        second = self.FakeDownloader("second")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [first, second]
        )

        result = CliRunner().invoke(
            app, ["datasets", "inspect", "--verbose"], prog_name="imp"
        )

        assert result.exit_code == 0, result.output
        assert first.inspect_calls == [True]
        assert second.inspect_calls == [True]

    def test_defaults_verbose_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default --verbose to False when not passed."""
        downloader = self.FakeDownloader("example")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [downloader]
        )

        result = CliRunner().invoke(app, ["datasets", "inspect"], prog_name="imp")

        assert result.exit_code == 0, result.output
        assert downloader.inspect_calls == [False]

    def test_logs_and_continues_past_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Log and continue with remaining downloaders when one fails to inspect."""
        failing = self.FakeDownloader("failing", error=RuntimeError("boom"))
        still_runs = self.FakeDownloader("still-runs")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders",
            lambda _config: [failing, still_runs],
        )

        with caplog.at_level(logging.ERROR):
            result = CliRunner().invoke(app, ["datasets", "inspect"], prog_name="imp")

        assert result.exit_code == 0, result.output
        assert "Inspecting dataset failing failed, skipping." in caplog.text
        assert still_runs.inspect_calls == [False]


class TestMasksCLI:
    class FakePostprocessor:
        """A minimal postprocessor stub exposing only what `masks` needs."""

        def __init__(self, *, error: Exception | None = None) -> None:
            """Store an optional error to raise from process()."""
            self.error = error
            self.process_calls: list[tuple[Path, bool]] = []

        def process(self, path_dataset: Path, *, overwrite: bool) -> None:
            """Record the call and raise the configured error, if any."""
            self.process_calls.append((path_dataset, overwrite))
            if self.error is not None:
                raise self.error

    class FakeDownloader:
        """A minimal downloader stub exposing only what `masks` needs."""

        def __init__(
            self, name: str, path_dataset: Path, postprocessor: object
        ) -> None:
            """Store the downloader's name, dataset path, and postprocessor."""
            self.name = name
            self.path_dataset = path_dataset
            self.postprocessor = postprocessor

    def test_calls_postprocessor_process_with_overwrite_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Call postprocessor.process(path_dataset, overwrite=True) per downloader."""
        postprocessor = self.FakePostprocessor()
        downloader = self.FakeDownloader(
            "example", tmp_path / "example.zarr", postprocessor
        )
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [downloader]
        )

        result = CliRunner().invoke(
            app, ["datasets", "masks", "--overwrite"], prog_name="imp"
        )

        assert result.exit_code == 0, result.output
        assert postprocessor.process_calls == [(downloader.path_dataset, True)]

    def test_defaults_overwrite_to_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default --overwrite to False when not passed."""
        postprocessor = self.FakePostprocessor()
        downloader = self.FakeDownloader(
            "example", tmp_path / "example.zarr", postprocessor
        )
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [downloader]
        )

        result = CliRunner().invoke(app, ["datasets", "masks"], prog_name="imp")

        assert result.exit_code == 0, result.output
        assert postprocessor.process_calls == [(downloader.path_dataset, False)]

    def test_propagates_errors_without_catching_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Let a postprocessor failure propagate uncaught, unlike create/inspect."""
        postprocessor = self.FakePostprocessor(error=RuntimeError("boom"))
        downloader = self.FakeDownloader(
            "example", tmp_path / "example.zarr", postprocessor
        )
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [downloader]
        )

        result = CliRunner().invoke(app, ["datasets", "masks"], prog_name="imp")

        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)
