# Add a new model

## Tensor format

All IceNet-MP models operate on tensors in `NTCHW` format:

| Dimension | Meaning |
|-----------|---------|
| `N` | Batch size |
| `T` | History steps (inputs) or forecast steps (outputs) |
| `C` | Channels / variables |
| `H` | Height |
| `W` | Width |

`N` and `T` are the same across all inputs, but `C`, `H`, and `W` may differ per dataset.

For example, with batch size `N=2`, 3 history steps, and 4 forecast steps, `k` inputs each have shape `(2, 3, C_k, H_k, W_k)` and the output has shape `(2, 4, C_out, H_out, W_out)`.

## Standalone models

A standalone model accepts a `dict[str, TensorNTCHW]` mapping dataset names to tensors and produces output of shape `(N, T, C_out, H_out, W_out)`. A separate model instance is typically needed for each output to be predicted.

![Standalone pipeline diagram](../assets/pipeline-standalone.png)

| | |
|---|---|
| **Pros** | All input variables available without transformation |
| **Cons** | Hard to add new inputs or outputs |

## Processor models

A processor model sits inside an encode-process-decode pipeline. You define a latent space `(C_latent, H_latent, W_latent)` — e.g. `(10, 64, 64)` — and the framework automatically creates one encoder per input and one decoder per output.

1. Each dataset-specific **encoder** maps input `(N, T, C_k, H_k, W_k)` → `(N, T, C_latent, H_latent, W_latent)`.
2. The `k` encoded tensors are concatenated to `(N, T, k·C_latent, H_latent, W_latent)`.
3. The **processor** maps `(N, T, k·C_latent, H_latent, W_latent)` → same shape.
4. Each output-specific **decoder** maps the processor output → `(N, T, C_out, H_out, W_out)`.

![Encode-process-decode pipeline diagram](../assets/pipeline-encode-process-decode.png)

| | |
|---|---|
| **Pros** | Easy to add new inputs or outputs |
| **Cons** | Input variables are transformed into latent space |
