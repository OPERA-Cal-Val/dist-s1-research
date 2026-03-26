import numpy as np
from matplotlib import pyplot as plt


def calculate_scalebar_size(profile, width_proportion=0.3):
    transform = profile["transform"]
    pixel_size = abs(transform[0])
    width_pixels = profile["width"]
    width_meters = width_pixels * pixel_size
    target_length = width_meters * width_proportion

    if target_length >= 1000:
        target = target_length / 1000
        units = "km"
    else:
        target = target_length
        units = "m"

    magnitude = 10 ** np.floor(np.log10(target))
    normalized = target / magnitude

    if normalized < 1.5:
        nice = 1
    elif normalized < 3.5:
        nice = 2
    elif normalized < 7.5:
        nice = 5
    else:
        nice = 10

    size = nice * magnitude
    return float(size), units


def add_scalebar(ax, profile, position="lower left", color="white", width_proportion=0.3, fontsize=10):
    position_map = {
        "upper left": (0.05, 0.95),
        "top left": (0.05, 0.95),
        "upper right": (0.95, 0.95),
        "top right": (0.95, 0.95),
        "lower left": (0.05, 0.05),
        "bottom left": (0.05, 0.05),
        "lower right": (0.95, 0.05),
        "bottom right": (0.95, 0.05),
    }

    x_frac, y_frac = position_map.get(position.lower(), (0.05, 0.05))

    size, units = calculate_scalebar_size(profile, width_proportion=width_proportion)

    if units == "km":
        size_meters = size * 1000
    else:
        size_meters = size

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]

    if x_frac > 0.5:
        x_end = xlim[0] + x_frac * width
        x_start = x_end - size_meters
    else:
        x_start = xlim[0] + x_frac * width
        x_end = x_start + size_meters

    y_pos = ylim[0] + y_frac * height

    ax.plot([x_start, x_end], [y_pos, y_pos], color=color, linewidth=3)

    text_x = (x_start + x_end) / 2
    if y_frac > 0.5:
        text_y = y_pos - 0.02 * height
        va = "top"
    else:
        text_y = y_pos + 0.02 * height
        va = "bottom"

    ax.text(
        text_x, text_y, f"{size:.0f} {units}", color=color, ha="center", va=va, fontsize=fontsize, weight="bold"
    )


def plot_scalebar(profile, position="lower left", figsize=(2, 0.5), color="white", width_proportion=0.3, fontsize=10):
    fig, ax = plt.subplots(figsize=figsize)

    transform = profile["transform"]
    width = profile["width"]
    height = profile["height"]

    left = transform[2]
    right = transform[2] + width * transform[0]
    top = transform[5]
    bottom = transform[5] + height * transform[4]

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.axis("off")

    add_scalebar(ax, profile, position=position, color=color, width_proportion=width_proportion, fontsize=fontsize)

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    return fig, ax
