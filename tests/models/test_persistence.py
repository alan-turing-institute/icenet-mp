import pytest
import torch

from icenet_mp.models import Persistence


class TestPersistence:
    @pytest.mark.parametrize("test_input_shape", [(512, 512, 4), (1000, 200, 1)])
    @pytest.mark.parametrize("test_output_shape", [(432, 432, 1), (10, 20, 19)])
    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 2, 5])
    @pytest.mark.parametrize("test_n_history_steps", [1, 2, 5])
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_input_shape: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_output_shape: tuple[int, int, int],
    ) -> None:
        input_space = {
            "channels": test_input_shape[2],
            "name": "input",
            "shape": test_input_shape[0:2],
        }
        output_space = {
            "channels": test_output_shape[2],
            "name": "target",
            "shape": test_output_shape[0:2],
        }
        model = Persistence(
            name="persistence",
            hemisphere="north",
            input_spaces=[input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=output_space,
            optimizer={},
            scheduler={},
        )
        batch = {
            "input": torch.randn(
                test_batch_size,
                test_n_history_steps,
                test_input_shape[2],
                test_input_shape[0],
                test_input_shape[1],
            ),
            "target": torch.randn(
                test_batch_size,
                test_n_forecast_steps,
                test_output_shape[2],
                test_output_shape[0],
                test_output_shape[1],
            ),
        }
        result: torch.Tensor = model(batch)
        assert result.shape == batch["target"].shape

    def test_optimizer(self) -> None:
        model = Persistence(
            name="persistence",
            hemisphere="north",
            input_spaces=[
                {
                    "channels": 1,
                    "name": "input",
                    "shape": (1, 1),
                }
            ],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space={
                "channels": 1,
                "name": "target",
                "shape": (1, 1),
            },
            optimizer={},
            scheduler={},
        )
        assert model.configure_optimizers() is None, (
            "No optimizer should be initialized"
        )
