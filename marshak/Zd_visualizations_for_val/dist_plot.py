from dist_s1.constants import DIST_STATUS_CMAP, DISTVAL2LABEL
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import Colorbar
from matplotlib.axes import Axes

DISTLABEL2CBARLABEL = {
    "nodata": "No Data",
    "no_disturbance": "No Dist",
    "first_low_conf_disturbance": "1st Low",
    "provisional_low_conf_disturbance": "Prov Low",
    "confirmed_low_conf_disturbance": "Conf Low",
    "first_high_conf_disturbance": "1st High",
    "provisional_high_conf_disturbance": "Prov High",
    "confirmed_high_conf_disturbance": "Conf High",
    "confirmed_low_conf_disturbance_finished": "Low Fin",
    "confirmed_high_conf_disturbance_finished": "High Fin",
}
DISTVAL2CBARLABEL = {k: DISTLABEL2CBARLABEL[v] for k, v in DISTVAL2LABEL.items()}


def get_dist_s1_mpl_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    values = sorted(DIST_STATUS_CMAP.keys())
    colors = np.array([DIST_STATUS_CMAP[v] for v in values]) / 255.0
    mpl_cmap = ListedColormap(colors)
    bounds = values + [values[-1] + 1]
    norm = BoundaryNorm(bounds, mpl_cmap.N)
    return mpl_cmap, norm


def get_colorbar_label(label: str, include_numerical_vals: bool = False) -> str:
    """Shorten label for compact display."""
    color_label_map = DISTVAL2CBARLABEL.copy()
    if include_numerical_vals:
        color_label_map = {k: f"{v} ({k})" for k, v in color_label_map.items()}
    return color_label_map[label]


def add_dist_s1_colorbar(cax: Axes, short_labels: bool = False, include_numerical_vals: bool = True) -> Colorbar:
    mpl_cmap, norm = get_dist_s1_mpl_cmap()
    values = sorted(DIST_STATUS_CMAP.keys())

    cb = Colorbar(cax, cmap=mpl_cmap, norm=norm)

    # Set tick positions at the center of each color bin
    tick_positions = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
    tick_positions.append(values[-1] + 0.5)
    cb.set_ticks(tick_positions)

    labels = [get_colorbar_label(val, include_numerical_vals=include_numerical_vals) for val in values]
    cb.set_ticklabels(labels)

    return cb
