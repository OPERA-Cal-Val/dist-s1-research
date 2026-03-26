import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import hsv_to_rgb, ListedColormap, Normalize, BoundaryNorm


def create_random_colormap_for_rasterio(
    n_colors: int = 256, seed: int | None = 0
) -> dict[int, tuple[int, int, int, int]]:
    assert n_colors <= 256, "n_colors must be <= 256"
    rng = np.random.default_rng(seed)
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


def create_random_colormap_matplotlib(
    n_colors: int = 256, background_val: int | None = None, seed: int | None = 0
) -> ListedColormap:
    rng = np.random.default_rng(seed)
    hsv = np.stack(
        [
            rng.random(n_colors),  # Full hue range
            rng.uniform(0.5, 1.0, n_colors),  # High saturation
            rng.uniform(0.4, 0.95, n_colors),  # Avoid very dark/light
        ],
        axis=1,
    )
    rgb = hsv_to_rgb(hsv)

    if background_val is not None:
        assert 0 <= background_val < n_colors, f"background_val must be between 0 and {n_colors - 1}"
        rgb[background_val] = [0, 0, 0]

    norm = Normalize(vmin=0, vmax=255)
    return ListedColormap(rgb), norm


def add_random_colors_colorbar(
    ax: Axes,
    cmap: ListedColormap | None = None,
    color_val_map: dict[int, str] | None = None,
    max_val: int = 255,
    background_val: int | None = None,
    fontsize: int = 15,
) -> Colorbar:
    if color_val_map is not None:
        assert all(0 <= v <= 255 for v in color_val_map.keys()), "All values in color_val_map must be between 0 and 255"

    if cmap is None:
        full_cmap, _ = create_random_colormap_matplotlib()
    else:
        full_cmap = cmap

    if background_val is not None:
        assert 0 <= background_val <= 255, "background_val must be between 0 and 255"
        colors = full_cmap.colors.copy()
        colors[background_val] = [0, 0, 0]
        full_cmap = ListedColormap(colors)

    if color_val_map is not None:
        values = sorted(color_val_map.keys())
        selected_colors = [full_cmap.colors[val] for val in values]
        colorbar_cmap = ListedColormap(selected_colors)
        boundaries = [v - 0.5 for v in values] + [values[-1] + 0.5]
        norm = BoundaryNorm(boundaries, len(values))
        tick_positions = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]
        labels = [color_val_map[val] for val in values]
    else:
        colorbar_cmap = full_cmap
        norm = Normalize(vmin=0, vmax=255)
        tick_positions = list(range(max_val + 1))
        labels = [str(val) for val in tick_positions]

    cb = Colorbar(ax, cmap=colorbar_cmap, norm=norm)
    cb.set_ticks(tick_positions)
    cb.set_ticklabels(labels, fontsize=fontsize)

    return cb


def add_continuous_colorbar_to_axes(
    ax: Axes, vmin: float, vmax: float, cmap: str = "viridis", label: str | None = None
) -> Colorbar:
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax)
    cb.set_ticks([vmin, vmax])
    if isinstance(vmin, float):
        cb.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    else:
        cb.set_ticklabels([f"{vmin}", f"{vmax}"])
    if label is not None:
        cb.set_label(label)
    return cb
