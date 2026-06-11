from __future__ import annotations

"""Figure 1: model geometry, gel network series, and control parameters.

Run:
    python scripts/fig1.py

Input:
    config/base.json
    data/structures/metrics.csv
    data/structures/g1.json ... g4.json
    data/timescales.csv

Output:
    figures/fig1/MM-DD-HH-MM.png   (timestamped, never overwritten)
"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


BLUE = "#1F5F99"
ORANGE = "#D99A2B"
GREEN = "#4C8C5A"
RED = "#B34444"
GRAY = "#6F6F6F"
LIGHT_BLUE = "#DCECF7"
LIGHT_GRAY = "#F5F5F5"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Liberation Serif",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def timestamp_name() -> str:
    now = datetime.now()
    return f"{now.month:02d}-{now.day:02d}-{now.hour:02d}-{now.minute:02d}"


def save_fig(fig: plt.Figure, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = timestamp_name()
    path = outdir / f"{stem}.png"
    while path.exists():
        stem = f"{stem}_1"
        path = outdir / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return path


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )


def structure_index(name: str) -> int:
    return int(name.lower().replace("g", ""))


def get_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def get_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def load_tau_ref(path: Path) -> float:
    if not path.exists():
        return float("nan")

    for row in read_csv(path):
        if row.get("structure", "").lower() == "g2" and row.get("tau_shape_used", ""):
            return float(row["tau_shape_used"])

    return float("nan")


def plot_pipe(ax: plt.Axes, radius: float, length: float) -> None:
    ax.axis("off")
    ax.set_aspect("equal")
    panel_label(ax, "A")

    # Cross-section.
    cx, cy = 0.0, 0.0
    ax.add_patch(Circle((cx, cy), 1.0, fill=False, lw=1.25, ec="black"))
    ax.add_patch(Circle((cx, cy), 0.33, fc=LIGHT_BLUE, ec=BLUE, lw=1.1))

    ax.annotate(
        "",
        xy=(1.0, 0.0),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", lw=1.0),
    )
    ax.text(0.53, 0.08, r"$R$", ha="center", va="bottom", fontsize=11)
    ax.text(0.0, -1.22, "cross-section", ha="center", va="top", fontsize=10)

    # Longitudinal section.
    x0, y0 = 2.05, -0.72
    w, h = 2.75, 1.44

    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, lw=1.25, ec="black"))
    ax.add_patch(Rectangle((x0 + 0.40, -0.22), w - 0.80, 0.44, fc=LIGHT_BLUE, ec=BLUE, lw=1.0))

    ax.annotate(
        "",
        xy=(x0 + w, y0 - 0.25),
        xytext=(x0, y0 - 0.25),
        arrowprops=dict(arrowstyle="<->", lw=1.0),
    )
    ax.text(x0 + w / 2, y0 - 0.39, r"$L$", ha="center", va="top", fontsize=11)

    y = np.linspace(-0.68, 0.68, 160)
    u = 1.0 - (y / 0.68) ** 2
    x = x0 + 0.43 + 0.58 * u
    ax.plot(x, y, color=RED, lw=2.0)

    ax.text(
        x0 + 1.04,
        0.55,
        r"$u_z(r)=u_{\max}\!\left(1-r^2/R^2\right)$",
        color=RED,
        ha="left",
        va="center",
        fontsize=10,
    )

    ax.annotate(
        "",
        xy=(x0 + w + 0.52, 0.0),
        xytext=(x0 + w + 0.08, 0.0),
        arrowprops=dict(arrowstyle="->", lw=1.1),
    )
    ax.text(x0 + w + 0.60, 0.0, r"$z$", ha="left", va="center", fontsize=11)

    ax.text(
        0.06,
        0.86,
        rf"$R={radius:g}$, $L={length:g}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )

    ax.text(
        0.50,
        1.02,
        "Cylindrical Poiseuille flow",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )

    ax.set_xlim(-1.35, 5.55)
    ax.set_ylim(-1.40, 1.25)


def plot_structure(ax: plt.Axes, structure: dict[str, Any], title: str) -> None:
    beads = structure["beads"]
    bonds = structure["bonds"]

    pos = {int(bead["id"]): np.asarray(bead["r"], dtype=float) for bead in beads}
    bead_type = {int(bead["id"]): str(bead["type"]) for bead in beads}

    pts = np.vstack(list(pos.values()))
    center = pts.mean(axis=0)
    span = np.max(np.ptp(pts[:, [0, 2]], axis=0))
    if span <= 0:
        span = 1.0

    def project(p: np.ndarray) -> tuple[float, float]:
        q = (p - center) / span
        return float(q[0]), float(q[2])

    for bond in bonds:
        i = int(bond["i"])
        j = int(bond["j"])
        x0, z0 = project(pos[i])
        x1, z1 = project(pos[j])
        ax.plot([x0, x1], [z0, z1], color="#AFAFAF", lw=0.65, zorder=1)

    chain_x, chain_z = [], []
    xlink_x, xlink_z = [], []

    for bead_id, p in pos.items():
        x, z = project(p)
        if bead_type[bead_id] == "xlink":
            xlink_x.append(x)
            xlink_z.append(z)
        else:
            chain_x.append(x)
            chain_z.append(z)

    ax.scatter(chain_x, chain_z, s=8, c=ORANGE, edgecolors="none", zorder=2)
    ax.scatter(xlink_x, xlink_z, s=30, c=BLUE, edgecolors="black", linewidths=0.35, zorder=3)

    ax.set_aspect("equal")
    ax.set_xlim(-0.64, 0.64)
    ax.set_ylim(-0.64, 0.64)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=4, fontsize=11)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#B0B0B0")


def plot_networks(fig: plt.Figure, gs_cell: Any, structures: dict[str, dict[str, Any]]) -> None:
    outer = gs_cell.subgridspec(2, 2, wspace=0.18, hspace=0.28)
    axes = [fig.add_subplot(outer[i, j]) for i in range(2) for j in range(2)]

    panel_label(axes[0], "B")

    for ax, name in zip(axes, ["g1", "g2", "g3", "g4"]):
        n = structure_index(name)
        plot_structure(ax, structures[name], rf"G{n}: ${n}\times{n}\times{n}$")

    axes[0].text(
        0.55,
        1.34,
        "shared-node network, x-z projection",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
    )


def plot_metrics(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    panel_label(ax, "C")

    names = [row["structure"].upper() for row in rows]
    x = np.arange(len(rows))

    rg = np.array([get_float(row, "Rg_over_R") for row in rows])
    r99 = np.array([get_float(row, "R99_over_R") for row in rows])
    n_bead = [get_int(row, "N_bead") for row in rows]
    n_cell = [structure_index(row["structure"]) ** 3 for row in rows]

    ax.plot(x, rg, marker="o", ms=6.5, lw=2.0, color=BLUE, label=r"$R_g/R$")
    ax.plot(x, r99, marker="s", ms=6.2, lw=2.0, color=GREEN, label=r"$R_{99}/R$")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            rf"{name}" + "\n" + rf"$N={n}$" + "\n" + rf"$N_{{cell}}={c}$"
            for name, n, c in zip(names, n_bead, n_cell)
        ],
        linespacing=1.15,
    )

    ymax = max(0.75, float(max(np.max(rg), np.max(r99))) * 1.25)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Relative size")
    ax.set_title("Gel size and network scale", pad=6)

    ax.legend(frameon=False, loc="upper left")
    ax.tick_params(which="both", top=True, right=True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def plot_control(ax: plt.Axes, config: dict[str, Any], tau_ref: float) -> None:
    ax.axis("off")
    panel_label(ax, "D")

    wi_values = config.get("flow", {}).get("wi_values", [0, 1, 3])
    wi_text = ", ".join(rf"${float(w):g}$" for w in wi_values)

    box = FancyBboxPatch(
        (0.06, 0.09),
        0.88,
        0.82,
        boxstyle="round,pad=0.026,rounding_size=0.025",
        transform=ax.transAxes,
        fc=LIGHT_GRAY,
        ec="#C9C9C9",
        lw=1.0,
    )
    ax.add_patch(box)

    if math.isfinite(tau_ref):
        tau_line = rf"$\tau_{{ref}}=\tau_{{shape}}(G2)={tau_ref:.2f}$"
    else:
        tau_line = r"$\tau_{ref}=\tau_{shape}(G2)$"

    text = "\n".join(
        [
            r"$\mathrm{Wi}_{ref}=\dot{\gamma}_{wall}\tau_{ref}$",
            tau_line,
            "",
            r"$\mathrm{Wi}_{s}=\dot{\gamma}_{wall}\tau_{shape,s}$",
            "",
            rf"planned $\mathrm{{Wi}}_{{ref}}$: {wi_text}",
            "",
            r"Pure-fluid calibration gives",
            r"$\dot{\gamma}_{wall}$ and $\mathrm{Re}_{R}$",
        ]
    )

    ax.text(
        0.13,
        0.82,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        linespacing=1.35,
    )

    ax.text(
        0.50,
        1.02,
        "Control parameters",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )


def build_figure(
    config: dict[str, Any],
    metrics_rows: list[dict[str, str]],
    structures: dict[str, dict[str, Any]],
    tau_ref: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(8.0, 6.4))

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.08, 1.00],
        height_ratios=[1.00, 1.00],
        left=0.065,
        right=0.985,
        bottom=0.085,
        top=0.970,
        wspace=0.24,
        hspace=0.34,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    plot_pipe(
        ax_a,
        radius=float(config["pipe"]["radius"]),
        length=float(config["pipe"]["length"]),
    )

    plot_networks(fig, gs[0, 1], structures)

    ax_c = fig.add_subplot(gs[1, 0])
    plot_metrics(ax_c, metrics_rows)

    ax_d = fig.add_subplot(gs[1, 1])
    plot_control(ax_d, config, tau_ref)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 1.")
    parser.add_argument("--config", type=Path, default=Path("config/base.json"))
    parser.add_argument("--metrics", type=Path, default=Path("data/structures/metrics.csv"))
    parser.add_argument("--timescales", type=Path, default=Path("data/timescales.csv"))
    parser.add_argument("--structures-dir", type=Path, default=Path("data/structures"))
    parser.add_argument("--output", type=Path, default=Path("figures/fig1"))
    args = parser.parse_args()

    root = Path.cwd()

    config = read_json(root / args.config)
    metrics_rows = sorted(
        read_csv(root / args.metrics),
        key=lambda row: structure_index(row["structure"]),
    )

    structures = {}
    for row in metrics_rows:
        name = row["structure"].lower()
        structures[name] = read_json(root / args.structures_dir / f"{name}.json")

    tau_ref = load_tau_ref(root / args.timescales)

    setup_style()
    fig = build_figure(config, metrics_rows, structures, tau_ref)
    out_path = save_fig(fig, root / args.output)

    print(f"Figure 1 written to: {out_path}")


if __name__ == "__main__":
    main()