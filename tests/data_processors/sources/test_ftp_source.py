"""Tests for the FTP data source."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from anemoi.datasets.create.recipe.dates import StartEndDates
from anemoi.datasets.dates.groups import GroupOfDates
from anemoi.utils.registry import Registry

from icenet_mp.data_processors.sources import FTPSource, register_sources


class TestFTPSource:
    """Test suite for FTPSource class."""

    mock_context = MagicMock()
    date_range = StartEndDates(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 3),
        frequency=timedelta(days=1),
    )
    dates = GroupOfDates(list(date_range), provider=date_range)

    def test_ftp_source_registration(self) -> None:
        """Test that FTPSource is properly registered."""
        # Mock source registry
        mock_registry = Registry("anemoi.datasets.create.sources")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "icenet_mp.data_processors.sources.source_registry",
                mock_registry,
            )
            assert "ftp" not in mock_registry.registered
            register_sources()
            assert "ftp" in mock_registry.registered
            assert mock_registry.lookup("ftp") == FTPSource

    def test_ftp_source_execute_basic(self) -> None:
        """Test basic FTP source execution with mocked FTP connection."""
        # Mock ftp.FTP
        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None

        # Mock ftp.load_one
        mock_load_one = MagicMock()
        mock_load_one.return_value = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("icenet_mp.data_processors.sources.ftp.FTP", mock_ftp_class)
            mp.setattr("icenet_mp.data_processors.sources.ftp.load_one", mock_load_one)

            # Execute
            source = FTPSource(
                context=self.mock_context,
                url=r"ftp://example.com/data/file.nc",
                user="testuser",
                passwd="testpass",  # noqa: S106
            )
            source.execute(dates=self.dates)

            # Verify FTP session was created with correct credentials
            mock_ftp_class.assert_called_once_with("example.com")
            mock_ftp.login.assert_called_once_with(user="testuser", passwd="testpass")  # noqa: S106

            # Verify load_one was called for each date
            assert mock_load_one.call_count == 3

    def test_ftp_source_execute_anonymous_login(self) -> None:
        """Test FTP source with anonymous login (default)."""
        # Mock ftp.FTP
        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None

        # Mock ftp.load_one
        mock_load_one = MagicMock()
        mock_load_one.return_value = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("icenet_mp.data_processors.sources.ftp.FTP", mock_ftp_class)
            mp.setattr("icenet_mp.data_processors.sources.ftp.load_one", mock_load_one)

            # Execute without providing user/passwd
            source = FTPSource(
                context=self.mock_context,
                url=r"ftp://example.com/data/file.nc",
            )
            source.execute(dates=self.dates)

            # Verify FTP session was created with correct credentials
            mock_ftp_class.assert_called_once_with("example.com")
            mock_ftp.login.assert_called_once_with(user="anonymous", passwd="")

            # Verify load_one was called for each date
            assert mock_load_one.call_count == 3

    def test_ftp_source_execute_file_download(self) -> None:
        """Test that URL parsing works correctly."""
        # Mock ftp.FTP
        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None

        # Mock ftp.load_one
        mock_load_one = MagicMock()
        mock_load_one.return_value = MagicMock()

        # Mock ftp.MultiFieldList
        mock_multi_field_list = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("icenet_mp.data_processors.sources.ftp.FTP", mock_ftp_class)
            mp.setattr("icenet_mp.data_processors.sources.ftp.load_one", mock_load_one)
            mp.setattr(
                "icenet_mp.data_processors.sources.ftp.MultiFieldList",
                mock_multi_field_list,
            )

            # Execute with a complex URL
            source = FTPSource(
                context=self.mock_context,
                url=r"ftp://data.server.com/archive/datasets/file.nc",
            )
            source.execute(dates=self.dates)

            # Verify correct server was used
            mock_ftp_class.assert_called_once_with("data.server.com")

            # Verify directory change was attempted
            mock_ftp.cwd.assert_called_with("/archive/datasets")
            assert mock_ftp.cwd.call_count == 3

            # Verify retrbinary was called to download files
            assert mock_ftp.retrbinary.call_args.args[0] == "RETR file.nc"
            assert mock_ftp.retrbinary.call_count == 3

            # Verify MultiFieldList was created with all downloaded files
            mock_multi_field_list.assert_called_once()

    def test_ftp_source_execute_pattern_substitution(self) -> None:
        """Test that date patterns are substituted correctly."""
        # Mock ftp.FTP
        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None

        # Mock ftp.load_one
        mock_load_one = MagicMock()
        mock_load_one.return_value = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("icenet_mp.data_processors.sources.ftp.FTP", mock_ftp_class)
            mp.setattr("icenet_mp.data_processors.sources.ftp.load_one", mock_load_one)

            source = FTPSource(
                context=self.mock_context,
                url=r"ftp://example.com/data/{date:strftime(%Y%m%d)}.nc",
            )
            source.execute(dates=self.dates)

            # Verify load_one was called with correct iso dates
            calls = mock_load_one.call_args_list
            assert len(calls) == 3

            # Check that iso dates are passed correctly
            assert "20200101.nc" in str(calls[0])
            assert "20200102.nc" in str(calls[1])
            assert "20200103.nc" in str(calls[2])
