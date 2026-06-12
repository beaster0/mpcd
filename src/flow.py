from __future__ import annotations

"""Run a short pure-fluid task, fit Poiseuille profile, save one PNG.

Usage:
    python src/flow.py
    python src/flow.py --profile results/r36_fluid_g0.0001_s301/profiles.npz
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run import read_json, run_dir, run_task


def timestamp_path(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%m-%d-%H-%M")
    path = outdir / f"{stem}.png"
    while path.exists():
        stem = f"{stem}_1"
        path = outdir / f"{stem}.png"
    return path


def fit_poiseuille(r: np.ndarray, vz: np.ndarray, radius: float) -> tuple[float, float]:
    mask = (r > 0) & (r < 0.95 * radius) & np.isfinite(vz)
    phi = 1.0 - (r[mask] / radius) ** 2
    y = vz[mask]
    u_max = float(np.dot(y, phi) / np.dot(phi, phi))
    fit = u_max * phi
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return u_max, r2


def summary_g(summary: dict) -> float:
    if "g" in summary:
        return float(summary["g"])
    body_force = summary.get("body_force")
    if isinstance(body_force, (list, tuple)) and len(body_force) >= 3:
        return float(body_force[2])
    return 0.0


def find_fluid_profile(root: Path, g: float) -> Path | None:
    for summary_path in sorted((root / "results").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("structure") != "fluid":
            continue
        if abs(summary_g(summary) - g) > 1e-15:
            continue
        profile = summary_path.parent / "profiles.npz"
        if profile.exists():
            return profile
    return None


def plot_profile(
    r: np.ndarray,
    vz: np.ndarray,
    radius: float,
    u_max: float,
    r2: float,
    g: float,
    label: str,
) -> Figure:
    r_line = np.linspace(0.0, radius, 200)
    vz_fit = u_max * (1.0 - (r_line / radius) ** 2)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(r, vz, "o", ms=5.5, mfc="#1F5F99", mec="white", mew=0.4, label="MPCD")
    ax.plot(r_line, vz_fit, "-", lw=2.0, color="#B34444", label=r"$u_{\max}(1-r^2/R^2)$")
    ax.axhline(0.0, color="#CCCCCC", lw=0.8, zorder=0)
    ax.set_xlim(0.0, radius)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\langle v_z \rangle$")
    ax.set_title("Poiseuille flow validation")
    ax.legend(frameon=False, loc="lower center")
    ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                rf"$g_z={g:g}$",
                rf"$u_{{\max}}={u_max:.4g}$",
                rf"$\dot{{\gamma}}_{{\mathrm{{wall}}}}={2 * u_max / radius:.4g}$",
                rf"$R^2={r2:.5f}$",
                label,
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F7F7F7", ec="#D0D0D0", lw=0.8),
    )
    ax.tick_params(which="both", top=True, right=True)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Poiseuille flow profile.")
    parser.add_argument("--config", type=Path, default=Path("config/base.json"))
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--g", type=float, default=0.0001)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--run-id", default="validate_flow")
    parser.add_argument("--no-run", action="store_true", help="only plot existing profile")
    parser.add_argument("--output", type=Path, default=Path("figures/flow"))
    args = parser.parse_args()

    root = Path.cwd()
    config = read_json(root / args.config)
    radius = float(config["pipe"]["radius"])
    profile = args.profile

    if profile is None and not args.no_run:
        task = {
            "run_id": args.run_id,
            "structure": "fluid",
            "g": str(args.g),
            "seed": str(args.seed),
            "steps": str(args.steps),
        }
        print(f"[run] {args.run_id} g={args.g} steps={args.steps}", flush=True)
        run_task(root, config, task, steps=args.steps)
        profile = run_dir(root, args.run_id) / "profiles.npz"
    elif profile is None:
        profile = find_fluid_profile(root, args.g)
        if profile is None:
            raise SystemExit("no fluid profile found; run without --no-run")

    data = np.load(profile)
    r = np.asarray(data["r"], dtype=float)
    vz = np.asarray(data["mean_vz"], dtype=float)
    u_max, r2 = fit_poiseuille(r, vz, radius)
    summary_path = profile.parent / "summary.json"
    if summary_path.exists():
        g = summary_g(json.loads(summary_path.read_text(encoding="utf-8")))
    else:
        g = args.g

    fig = plot_profile(r, vz, radius, u_max, r2, g, profile.parent.name)
    out = timestamp_path(root / args.output)
    fig.savefig(out, bbox_inches="tight", dpi=600)
    plt.close(fig)

    print(f"g={g:g}  u_max={u_max:.6g}  gamma_wall={2 * u_max / radius:.6g}  R2={r2:.6f}")
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
