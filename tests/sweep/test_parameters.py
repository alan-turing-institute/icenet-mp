import pytest

from icenet_mp.sweep.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    build_parameter,
)


class TestIntParameter:
    """Tests for IntParameter."""

    def test_accepts_log_with_default_step(self) -> None:
        parameter = IntParameter("x", 1, 10, log=True)
        assert parameter.step == 1

    def test_accepts_a_non_default_step_without_log(self) -> None:
        parameter = IntParameter("x", 1, 10, step=2)
        assert parameter.step == 2

    def test_rejects_log_with_a_non_default_step(self) -> None:
        with pytest.raises(ValueError, match=r"step.*must be 1"):
            IntParameter("x", 1, 10, log=True, step=2)

    def test_from_spec_rejects_log_with_a_non_default_step(self) -> None:
        spec = {"low": 1, "high": 10, "log": True, "step": 2}
        with pytest.raises(ValueError, match=r"step.*must be 1"):
            IntParameter.from_spec("x", spec)

    def test_sweep_cfg_uses_int_uniform_by_default(self) -> None:
        cfg = IntParameter("x", 1, 10).sweep_cfg()
        assert cfg["x"] == {"distribution": "int_uniform", "min": 1, "max": 10}

    def test_sweep_cfg_quantizes_a_non_default_step(self) -> None:
        cfg = IntParameter("x", 1, 10, step=2).sweep_cfg()
        assert cfg["x"] == {
            "distribution": "q_uniform",
            "min": 1,
            "max": 10,
            "q": 2,
        }

    def test_sweep_cfg_sets_q_to_one_when_log(self) -> None:
        cfg = IntParameter("x", 1, 10, log=True).sweep_cfg()
        assert cfg["x"] == {
            "distribution": "q_log_uniform_values",
            "min": 1,
            "max": 10,
            "q": 1,
        }


class TestFloatParameter:
    """Tests for FloatParameter."""

    def test_accepts_log_without_a_step(self) -> None:
        parameter = FloatParameter("x", 0.1, 10.0, log=True)
        assert parameter.step is None

    def test_accepts_a_step_without_log(self) -> None:
        parameter = FloatParameter("x", 0.1, 10.0, step=0.5)
        assert parameter.step == 0.5

    def test_rejects_log_with_a_step(self) -> None:
        with pytest.raises(ValueError, match=r"step.*not supported"):
            FloatParameter("x", 0.1, 10.0, log=True, step=0.5)

    def test_from_spec_rejects_log_with_a_step(self) -> None:
        spec = {"low": 0.1, "high": 10.0, "log": True, "step": 0.5}
        with pytest.raises(ValueError, match=r"step.*not supported"):
            FloatParameter.from_spec("x", spec)

    def test_sweep_cfg_uses_uniform_by_default(self) -> None:
        cfg = FloatParameter("x", 0.1, 10.0).sweep_cfg()
        assert cfg["x"] == {"distribution": "uniform", "min": 0.1, "max": 10.0}

    def test_sweep_cfg_quantizes_a_step(self) -> None:
        cfg = FloatParameter("x", 0.1, 10.0, step=0.5).sweep_cfg()
        assert cfg["x"] == {
            "distribution": "q_uniform",
            "min": 0.1,
            "max": 10.0,
            "q": 0.5,
        }

    def test_sweep_cfg_stays_unquantized_when_log(self) -> None:
        cfg = FloatParameter("x", 0.1, 10.0, log=True).sweep_cfg()
        assert cfg["x"] == {
            "distribution": "log_uniform_values",
            "min": 0.1,
            "max": 10.0,
        }


class TestBuildParameter:
    """Tests for build_parameter."""

    def test_builds_an_int_parameter(self) -> None:
        parameter = build_parameter("x", {"type": "int", "low": 1, "high": 10})
        assert isinstance(parameter, IntParameter)

    def test_builds_a_float_parameter(self) -> None:
        parameter = build_parameter("x", {"type": "float", "low": 0.1, "high": 10.0})
        assert isinstance(parameter, FloatParameter)

    def test_builds_a_categorical_parameter(self) -> None:
        parameter = build_parameter("x", {"type": "categorical", "choices": [1, 2]})
        assert isinstance(parameter, CategoricalParameter)

    def test_raises_for_an_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown parameter type"):
            build_parameter("x", {"type": "not-a-type"})
