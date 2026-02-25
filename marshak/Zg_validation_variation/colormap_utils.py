import numpy as np
from matplotlib.colors import hsv_to_rgb


def create_random_colormap(n_colors: int = 256) -> dict[int, tuple[int, int, int, int]]:
    assert n_colors <= 256, "n_colors must be <= 256"
    rng = np.random.default_rng()
    hsv = np.stack(
        [
            rng.random(n_colors),  # Full hue range
            rng.uniform(0.5, 1.0, n_colors),  # High saturation
            rng.uniform(0.4, 0.95, n_colors),  # Avoid very dark/light
        ],
        axis=1,
    )
    rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
    return {i: (*color, 255) for i, color in enumerate(rgb)}
