# OUTDATED
# Legacy diagram generator for earlier model framing. Final SCUC proxy reporting
# is captured in final_scuc_proxy_results.md and figures/final_scuc_proxy.
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BOX = dict(boxstyle="round,pad=0.25,rounding_size=0.05", linewidth=1.4, edgecolor="#1f2933")


def _box(ax, xy, w, h, text, fc="#f8fafc", fontsize=10):
    patch = FancyBboxPatch(xy, w, h, facecolor=fc, **BOX)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, start, end, text=None, dy=0.0):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#334155")
    ax.add_patch(arr)
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + dy, text, ha="center", va="center", fontsize=8)


def residual_soft_repair(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Residual Soft-Repair Dispatch Pipeline", fontsize=18, pad=18)

    _box(ax, (0.3, 4.3), 2.2, 1.0, "Causal NYISO Inputs\nforecast load, DA prices,\nweather, outages, lags", "#dbeafe")
    _box(ax, (3.0, 4.3), 2.2, 1.0, "24-hour Feature Window\nX_t = [x_{t-23}, ..., x_t]", "#e0f2fe")
    _box(ax, (5.7, 4.3), 2.2, 1.0, "Neural Network\npredicts dispatch change\nDelta g_hat_t", "#dcfce7")
    _box(ax, (8.4, 4.3), 2.0, 1.0, "Raw Dispatch\ng_raw = g_{t-1} + Delta g_hat", "#fef9c3")
    _box(ax, (10.8, 4.3), 1.9, 1.0, "Soft Repair LP\nfinal dispatch g*_t", "#fae8ff")

    _arrow(ax, (2.5, 4.8), (3.0, 4.8))
    _arrow(ax, (5.2, 4.8), (5.7, 4.8))
    _arrow(ax, (7.9, 4.8), (8.4, 4.8))
    _arrow(ax, (10.4, 4.8), (10.8, 4.8))

    _box(ax, (3.0, 2.2), 2.2, 0.9, "Training Target\nDelta g_t = y_t - y_{t-1}", "#fee2e2")
    _box(ax, (5.7, 2.2), 2.2, 0.9, "Supervised Loss\n||Delta g_hat - Delta g||²", "#fee2e2")
    _arrow(ax, (4.1, 3.1), (6.8, 4.3), "train NN", dy=-0.25)
    _arrow(ax, (6.8, 2.2), (6.8, 4.3), "backprop to NN", dy=0.0)

    _box(
        ax,
        (8.35, 1.05),
        4.35,
        1.45,
        "Repair LP Objective and Constraints\nmin |g - g_raw| + p_balance(balance deviation)\ncapacity + empirical ramp bounds + soft balance",
        "#f3e8ff",
        fontsize=9,
    )
    _arrow(ax, (9.4, 4.3), (10.2, 2.5), "post-training repair", dy=0.1)

    ax.text(
        0.35,
        0.35,
        "Interpretation: the neural model learns hourly fuel-mix changes; the LP repairs the raw prediction into a dispatch-like decision.",
        fontsize=10,
        color="#334155",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def residual_aware_lto(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_title("Residual-Aware Fenchel-Young Learning-to-Optimize Pipeline", fontsize=18, pad=18)

    _box(ax, (0.3, 4.4), 2.2, 1.0, "Causal NYISO Inputs\nforecast load, DA prices,\nweather, outages, lags", "#dbeafe")
    _box(ax, (3.0, 4.4), 2.2, 1.0, "24-hour Feature Window\nX_t", "#e0f2fe")
    _box(ax, (5.7, 4.4), 2.2, 1.0, "Neural Network\nlearns adjustment costs\nc_up, c_down", "#dcfce7")
    _box(ax, (8.4, 4.4), 2.0, 1.0, "Residual-Aware LP\ng = g_{t-1} + u - d", "#fae8ff")
    _box(ax, (10.8, 4.4), 1.9, 1.0, "Final Dispatch\ng*_t", "#fef9c3")

    _arrow(ax, (2.5, 4.9), (3.0, 4.9))
    _arrow(ax, (5.2, 4.9), (5.7, 4.9))
    _arrow(ax, (7.9, 4.9), (8.4, 4.9))
    _arrow(ax, (10.4, 4.9), (10.8, 4.9))

    _box(ax, (0.7, 2.25), 2.45, 0.95, "P-63 Oracle y_t\nhistorical fuel mix", "#fee2e2")
    _box(ax, (3.7, 2.25), 2.45, 0.95, "Projected Oracle y_proj\nclosest feasible easy solution", "#fee2e2")
    _box(ax, (7.1, 2.1), 2.9, 1.25, "Fenchel-Young Signal\nu_proj - u_pred\nd_proj - d_pred", "#ffedd5")
    _box(ax, (10.5, 2.25), 2.0, 0.95, "Backprop\nupdates NN costs", "#dcfce7")

    _arrow(ax, (3.15, 2.72), (3.7, 2.72), "projection into easy feasible set", dy=0.22)
    _arrow(ax, (6.15, 2.72), (7.1, 2.72))
    _arrow(ax, (10.0, 2.72), (10.5, 2.72))
    _arrow(ax, (11.5, 3.2), (6.8, 4.4), "end-to-end learning signal", dy=0.0)
    _arrow(ax, (9.4, 4.4), (8.55, 3.35), "LP solution\nu_pred, d_pred", dy=-0.1)

    _box(
        ax,
        (4.5, 0.7),
        4.5,
        1.0,
        "Easy Problem\nmin c_up u + c_down d + p_balance balance deviation\nsubject to capacity, ramping, and soft balance",
        "#f3e8ff",
        fontsize=9,
    )

    ax.text(
        0.35,
        0.25,
        "Interpretation: the neural model learns parameters of an optimization problem; the LP solution drives the structured loss.",
        fontsize=10,
        color="#334155",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    out = Path("figures")
    out.mkdir(parents=True, exist_ok=True)
    residual_soft_repair(out / "diagram_residual_soft_repair.png")
    residual_aware_lto(out / "diagram_residual_aware_fy_lto.png")
    print("Saved diagrams:")
    print(out / "diagram_residual_soft_repair.png")
    print(out / "diagram_residual_aware_fy_lto.png")


if __name__ == "__main__":
    main()
