"""Shared visual constants for all analysis figures.

Import from here to keep report figures visually consistent.
"""

import matplotlib.colors as mcolors

H_L = 0.2
H_U = 0.8

# Three-zone colormap: deep blue below H_L, neutral cream in viable range,
# deep red above H_U.  Luminance profile dark–light–dark survives greyscale.
FIRING_RATE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "hp_activity",
    [
        (0.000, "#1A4F8A"),   # deep blue  — under-activity
        (0.199, "#1A4F8A"),   # holds to H_L boundary
        (0.200, "#F0EFE8"),   # light cream — start of viable range
        (0.800, "#F0EFE8"),   # light cream — end of viable range
        (0.801, "#A61C22"),   # deep red   — start of over-activity
        (1.000, "#A61C22"),   # holds to top
    ],
)

# Zone colours for line plots, sampled from the three cmap regions
COLOR_UNDER  = "#1A4F8A"   # deep blue  — U (under-active)
COLOR_VIABLE = "#F0EFE8"   # light cream — V (viable); invisible on white, use line variants below
COLOR_OVER   = "#A61C22"   # deep red   — O (over-active)

# Line-plot colours: visible on white backgrounds where cream (#F0EFE8) disappears
COLOUR_VIABLE_LINE = "#2ca02c"   # mid-green   — V viable fraction / dwell / entry-exit
COLOUR_UNDER_LINE  = "#1A4F8A"   # deep blue   — U under-active; matches heatmap zone
COLOUR_OVER_LINE   = "#A61C22"   # deep red    — O over-active;  matches heatmap zone

CONDITION_ORDER: list[str] = ["HP_OFF", "HP_DEV_ONLY", "HP_BEHAVIOUR_ONLY", "HP_BOTH"]
CONDITION_LABELS: dict[str, str] = {
    "HP_OFF":            "No HP",
    "HP_DEV_ONLY":       "Dev only",
    "HP_BEHAVIOUR_ONLY": "Behaviour only",
    "HP_BOTH":           "Both",
}

NEURON_LABELS: list[str] = [
    "Left sensor",
    "Centre sensor",
    "Right sensor",
    "Left motor",
    "Right motor",
]
