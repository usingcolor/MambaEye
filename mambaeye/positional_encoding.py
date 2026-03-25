import torch


def sinusoidal_position_encoding_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Computes standard sinusoidal positional encoding for 1D positions.

    Args:
        positions: Tensor of shape (...,) containing scalar positions.
        dim: Even integer, the embedding dimensionality for this single axis.

    Returns:
        Tensor of shape (..., dim) with sin/cos features.
    """
    if dim % 2 != 0:
        raise ValueError("dim must be even for sinusoidal_position_encoding_1d")

    original_shape = positions.shape
    positions = positions.float().unsqueeze(-1)  # (..., 1)

    half_dim = dim // 2
    device = positions.device
    # Exponents scaled as in Vaswani et al. (2017): 10000^(2i/d)
    exponent = torch.arange(half_dim, device=device, dtype=torch.float32)
    div_term = torch.pow(10000.0, (2 * exponent) / dim)  # (half_dim,)

    angles = positions / div_term  # (..., half_dim)
    sin_vals = torch.sin(angles)
    cos_vals = torch.cos(angles)
    pe = torch.cat([sin_vals, cos_vals], dim=-1)  # (..., dim)
    return pe.view(*original_shape, dim)


def sinusoidal_position_encoding_2d(positions_xy: torch.Tensor, dim_per_axis: int) -> torch.Tensor:
    """Computes 2D sinusoidal positional encoding from (x, y) positions.

    Args:
        positions_xy: Tensor of shape (..., 2) containing [x, y] positions. Values can be any scale.
        dim_per_axis: Even integer, the embedding dimension per axis. The output dimension is 2*dim_per_axis.

    Returns:
        Tensor of shape (..., 2*dim_per_axis) representing concatenated encodings for x and y axes.
    """
    if positions_xy.shape[-1] != 2:
        raise ValueError("positions_xy must have shape (..., 2)")
    if dim_per_axis % 2 != 0:
        raise ValueError("dim_per_axis must be even for sinusoidal_position_encoding_2d")

    x = positions_xy[..., 0]
    y = positions_xy[..., 1]
    pe_x = sinusoidal_position_encoding_1d(x, dim_per_axis)
    pe_y = sinusoidal_position_encoding_1d(y, dim_per_axis)
    return torch.cat([pe_x, pe_y], dim=-1)


