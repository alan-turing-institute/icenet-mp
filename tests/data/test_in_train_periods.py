import numpy as np

from icenet_mp.data.climatology import in_train_periods


class TestInTrainPeriods:
    """`in_train_periods` decides which dates the climatology average may include.

    `build_climatology` itself is covered end-to-end (real dates, real averaging,
    the 29 February fallback, error cases) by tests/data/test_climatology_data.py
    against a real zarr fixture; these tests isolate just the period-membership
    check, which needs no I/O.
    """

    def test_date_within_a_bounded_period_is_included(self) -> None:
        periods = [{"start": "2018-01-01", "end": "2018-12-31"}]
        assert in_train_periods(np.datetime64("2018-06-15"), periods)

    def test_date_outside_every_period_is_excluded(self) -> None:
        periods = [{"start": "2018-01-01", "end": "2018-12-31"}]
        assert not in_train_periods(np.datetime64("2019-01-01"), periods)

    def test_date_matches_any_one_of_several_periods(self) -> None:
        periods = [
            {"start": "2017-01-01", "end": "2017-12-31"},
            {"start": "2019-01-01", "end": "2019-12-31"},
        ]
        assert in_train_periods(np.datetime64("2019-06-15"), periods)

    def test_unbounded_start_only_checks_the_end_bound(self) -> None:
        periods = [{"start": None, "end": "2018-12-31"}]
        assert in_train_periods(np.datetime64("2000-01-01"), periods)
        assert not in_train_periods(np.datetime64("2019-01-01"), periods)

    def test_unbounded_end_only_checks_the_start_bound(self) -> None:
        periods = [{"start": "2018-01-01", "end": None}]
        assert in_train_periods(np.datetime64("2030-01-01"), periods)
        assert not in_train_periods(np.datetime64("2017-12-31"), periods)

    def test_fully_unbounded_period_includes_every_date(self) -> None:
        periods = [{"start": None, "end": None}]
        assert in_train_periods(np.datetime64("1900-01-01"), periods)

    def test_bounds_are_compared_at_day_precision(self) -> None:
        """A bound with a time component behaves like its calendar day."""
        periods = [{"start": "2019-01-01T12:00:00", "end": "2019-01-01T12:00:00"}]
        assert in_train_periods(np.datetime64("2019-01-01T00:00:00"), periods)

    def test_no_periods_excludes_every_date(self) -> None:
        assert not in_train_periods(np.datetime64("2019-01-01"), [])
