import math
import random
from typing import List


def ensure_positions_cover_sequence(
    positions: List[tuple[int, int]],
    sequence_length: int,
) -> List[tuple[int, int]]:
    if not positions:
        return [(0, 0)] * sequence_length
    if len(positions) >= sequence_length:
        return positions[:sequence_length]
    # Repeat positions to match sequence_length
    repeat_factor = math.ceil(sequence_length / len(positions))
    extended = (positions * repeat_factor)[:sequence_length]
    return extended


def generate_scan_positions(
    x_start: int,
    x_stop: int,
    y_start: int,
    y_stop: int,
    patch_size: int,
    sequence_length: int,
    scan_pattern: str,
    rng: random.Random,
) -> List[tuple[int, int]]:
    """
    Generate a list of (x, y) patch top-left coordinates based on the scan pattern.
    Supports: column_major, horizontal_raster, horizontal_zigzag, horizontal_snake,
              column_snake, vertical_raster, hilbert, spiral, diagonal, golden,
              random_fixed_grid, random.
    """
    stride = patch_size

    xs = list(range(x_start, max(x_stop, x_start + 1), stride))
    ys = list(range(y_start, max(y_stop, y_start + 1), stride))

    if xs and xs[-1] > x_stop - patch_size:
        xs[-1] = max(x_start, x_stop - patch_size)
    if ys and ys[-1] > y_stop - patch_size:
        ys[-1] = max(y_start, y_stop - patch_size)

    nx, ny = len(xs), len(ys)
    positions: List[tuple[int, int]] = []

    if scan_pattern == "column_major":
        for xi in xs:
            for yi in ys:
                positions.append((xi, yi))
    elif scan_pattern == "horizontal_raster":
        for yi in ys:
            for xi in xs:
                positions.append((xi, yi))
    elif scan_pattern in ("horizontal_zigzag", "horizontal_snake"):
        # Traverse rows; alternate direction per row for a zig-zag pattern.
        for row_index, yi in enumerate(ys):
            if row_index % 2 == 0:
                for xi in xs:
                    positions.append((xi, yi))
            else:
                for xi in reversed(xs):
                    positions.append((xi, yi))
    elif scan_pattern in ("column_snake", "vertical_raster"):
        for col_index, xi in enumerate(xs):
            if col_index % 2 == 0:
                for yi in ys:
                    positions.append((xi, yi))
            else:
                for yi in reversed(ys):
                    positions.append((xi, yi))
    elif scan_pattern == "hilbert":
        # side length n must be a power of 2 for Hilbert curve
        m = max(nx, ny)
        n_pow2 = 1
        while n_pow2 < m:
            n_pow2 *= 2

        def d2xy(n, d):
            t, x, y, s = d, 0, 0, 1
            while s < n:
                rx = 1 & (t // 2)
                ry = 1 & (t ^ rx)
                if ry == 0:
                    if rx == 1:
                        x, y = s - 1 - x, s - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
                t //= 4
                s *= 2
            return x, y

        for d in range(n_pow2 * n_pow2):
            ix, iy = d2xy(n_pow2, d)
            if ix < nx and iy < ny:
                positions.append((xs[ix], ys[iy]))
    elif scan_pattern == "spiral":
        visited = [[False] * ny for _ in range(nx)]
        ix, iy = 0, 0
        # Directions: Right, Down, Left, Up
        dxs, dys = [1, 0, -1, 0], [0, 1, 0, -1]
        di = 0
        for _ in range(nx * ny):
            positions.append((xs[ix], ys[iy]))
            visited[ix][iy] = True
            nix, niy = ix + dxs[di], iy + dys[di]
            if 0 <= nix < nx and 0 <= niy < ny and not visited[nix][niy]:
                ix, iy = nix, niy
            else:
                di = (di + 1) % 4
                ix, iy = ix + dxs[di], iy + dys[di]
                if not (0 <= ix < nx and 0 <= iy < ny and not visited[ix][iy]):
                    break
    elif scan_pattern == "diagonal":
        for s in range(nx + ny - 1):
            for ix in range(max(0, s - ny + 1), min(nx, s + 1)):
                iy = s - ix
                positions.append((xs[ix], ys[iy]))
    elif scan_pattern == "golden":
        # Deterministic global coverage using Golden Ratio stride
        # Generates a low-discrepancy sequence over the grid patches
        base_positions = []
        for yi in ys:
            for xi in xs:
                base_positions.append((xi, yi))
        
        n = len(base_positions)
        if n > 1:
            # 0.618033... is (phi - 1)
            stride = int(n * 0.618033988749895)
            # Ensure stride is coprime to n to visit all points
            if stride < 1:
                stride = 1
            while math.gcd(stride, n) != 1:
                stride += 1
            
            positions = [base_positions[(i * stride) % n] for i in range(n)]
        else:
            positions = base_positions
    elif scan_pattern == "random_fixed_grid":
        for yi in ys:
            for xi in xs:
                positions.append((xi, yi))
        rng.shuffle(positions)
    elif scan_pattern == "random":
        # Completely random with 1 pixel unit, allowing overlay.
        # Sample 'sequence_length' positions from the valid range, allowing replacements.
        max_x = max(x_start, x_stop - patch_size)
        max_y = max(y_start, y_stop - patch_size)

        # We need exactly sequence_length positions
        for _ in range(sequence_length):
            rx = rng.randint(x_start, max_x)
            ry = rng.randint(y_start, max_y)
            positions.append((rx, ry))
    else:
        raise ValueError(f"Unknown scan pattern: {scan_pattern}")

    return ensure_positions_cover_sequence(positions, sequence_length)
