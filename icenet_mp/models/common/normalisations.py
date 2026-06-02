from torch import nn


def normalisation_from_name(norm_type: str, out_channels: int) -> nn.Module:
    """Create a normalization layer based on the specified type."""
    # Batch normalisation
    if norm_type.lower() == "batchnorm":
        return nn.BatchNorm2d(out_channels)

    # Group normalisation
    if norm_type.lower() == "groupnorm":
        # Determine the highest integer less than 8 that divides `channels`
        num_groups = max(num for num in range(1, 9) if out_channels % num == 0)
        return nn.GroupNorm(num_groups, out_channels)

    # No normalisation
    if norm_type.lower() == "none":
        return nn.Identity()

    msg = f"Unknown norm_type: {norm_type}. Choose 'groupnorm', 'batchnorm', or 'none'"
    raise ValueError(msg)
