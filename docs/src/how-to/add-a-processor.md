# Add a processor

A processor sits between the encoders and decoder in the encode-process-decode pipeline.
It receives the concatenated latent representations of all inputs and produces a latent forecast.

## The processor interface

All IceNet-MP processors extend `BaseProcessor` from `icenet_mp.models.processors`.
They operate on tensors in `NTCHW` format, taking in a tensor with a number of history steps and returning a tensor with a number of forecast steps.
For example, with 3 history steps, and 4 forecast steps, a processor will convert a tensor of shape `(N, 3, C, H, W)` to `(N, 4, C, H, W)`

The base class exposes two entry points, and you only need to implement one:

| Method | Signature | When to override |
|--------|-----------|-----------------|
| `forward` | `(x: TensorNCHW) -> TensorNCHW` | Stateless single-timestep transforms |
| `rollout` | `(x: TensorNTCHW, y: TensorNTCHW \| None) -> ModelStepOutput` | Any model that needs access to the full temporal history, or that behaves differently during training vs. inference |

The default `rollout` implementation calls `forward` once per forecast step, passing each prediction back as the next input.
If your architecture works on one timestep at a time and uses the same logic during training and inference, only overriding `forward` is sufficient.

## Simple processor: override `forward`

```python
from typing import Any
from icenet_mp.models.processors import BaseProcessor
from icenet_mp.types import TensorNCHW


class MyProcessor(BaseProcessor):
    def __init__(self, *, hidden_dim: int = 128, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        in_channels = self.data_space.channels
        self.model = ...  # your nn.Module here

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
```

This model can be trained in either single-stage or multistage mode.


## Training vs. inference: override `rollout`

Some architectures fundamentally differ between training and inference.
The canonical example is a diffusion model: during training you corrupt the target and predict noise; during inference you run the full reverse diffusion chain from pure noise.

If you use the multistage training flow - encode and decode components can be pretrained independently before the processor is trained on their fixed latent space.
This then allows the use of different training and inference behaviour in the `rollout` method.

The `rollout` signature allows the processor to handle both training and inference without direct knowledge of which step is being run:

- if `y`, the latent-space-encoded target, is provided, this is **training**
- if `y` is `None` then this is **inference**

```python
from torch import Tensor
from icenet_mp.models.processors import BaseProcessor
from icenet_mp.types import ModelStepOutput, TensorNTCHW


class MyDiffusionProcessor(BaseProcessor):

    def rollout(
        self, x: TensorNTCHW, y: TensorNTCHW | None = None
    ) -> ModelStepOutput:
        # x: (N, T_history, C, H, W) - encoded inputs
        # y: (N, T_forecast, C, H, W) - encoded targets
        if y is not None:
            # --- Training path ---
            prediction, loss = self._training(x, y)
            return ModelStepOutput(prediction=prediction, target=y, loss=loss)
        else:
            # --- Inference path ---
            prediction = self._inference(x)
            return ModelStepOutput(prediction=prediction, target=None, loss=None)
```

Returning a valid `loss` tensor tells `ProcessorStage` to skip its own loss computation and use yours instead.
The decoded prediction is still computed and logged, but gradients flow through your custom loss.

## Register the processor in config

Add a model config under `icenet_mp/config/model/` that points `processor._target_` at your class:

```yaml
# icenet_mp/config/model/cnn_mydiffusion_cnn.yaml
_target_: icenet_mp.models.EncodeProcessDecode

name: cnn-ddpm-cnn

encoders:
  latent_space: [144, 144]
  era5:
    _target_: icenet_mp.models.encoders.CNNEncoder
  sic-icenet:
    _target_: icenet_mp.models.encoders.CNNEncoder

processor:
  _target_: icenet_mp.models.processors.MyDiffusionProcessor
  timesteps: 1000

decoder:
  _target_: icenet_mp.models.decoders.CNNDecoder
  bounded: false
```

Then run training with:

```bash
uv run imp train model=cnn_mydiffusion_cnn
```
