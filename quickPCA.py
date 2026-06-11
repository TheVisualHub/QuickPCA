# quick_pca.py (ver 1.00)
# Essential Dynamics Analysis for MD trajecotories in PyMOL.
# Author: Gleb Novikov
# © The Visual Hub 2026
# For educational use only. 
# If you use QuickPCA in your research, please cite this tool.
# Any contributions toward its development are also appreciated.
#
# INSTRUCTIONS:
# 1. Place this script in the same folder as your topology (pdb) and trajectory files.
# 2. Drag-and-drop the script into the PyMOL window — it runs automatically.
#    Supported trajectory formats: .nc  .xtc  .trr  .dcd
#
#
# Output: PCA_Report.png  (Free-Energy Landscape + explained-variance chart
#                          + cross-correlation matrix + PC1/PC2 projections)
#
# Dependencies: numpy, scikit-learn, scipy, matplotlib
#   pip install numpy scikit-learn scipy matplotlib

import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from pymol import cmd

# =============================================================================
# ⚙️  USER SETTINGS  — edit these before running
# =============================================================================

# SVD performed directly on (n_frames × 3N) Cα coordinate matrix
PCA_SEL    = "polymer and name CA"

# Number of principal components to compute (≥2 required)
PCA_NCOMP  = 10

# Free-Energy Landscape histogram resolution and smoothing
PCA_NBINS  = 50      # bins per axis
PCA_SIGMA  = 1.0     # Gaussian σ in bin units

# Temperature (Kelvin) for Boltzmann inversion
PCA_TEMP   = 300.0

# Input trajectory options
MD_INTERVAL = 5 # takes every 5 snapshots from initial data

# Output filename
OUTPUT_PNG = "PCA_Report.png"

# =============================================================================
# 🔬  CORE FUNCTIONS
# =============================================================================

def compute_pca(obj_name, selection=PCA_SEL, n_components=PCA_NCOMP):
    """
    Extract Cα coordinates from every PyMOL state, Kabsch-align each frame to
    the first, mean-centre, and run sklearn PCA.

    Returns a result dict (or None on failure).
    """
    try:
        from sklearn.decomposition import PCA as _PCA
    except ImportError:
        print("❌  scikit-learn not found.  pip install scikit-learn")
        return None

    n_states = cmd.count_states(obj_name)
    if n_states < 3:
        print(f"❌  Need at least 3 frames, found {n_states}.")
        return None

    print(f"   Extracting '{selection}' from {n_states} states …")

    ref_pos = ref_com = None
    frames  = []

    for state in range(1, n_states + 1):
        model  = cmd.get_model(f"({obj_name}) and ({selection})", state=state)
        coords = np.array([a.coord for a in model.atom], dtype=np.float64)

        if coords.size == 0:
            print(f"   ⚠️  State {state}: no atoms matched — skipping.")
            continue

        # ── Kabsch alignment to frame 1 ──────────────────────────────────────
        if ref_pos is None:
            ref_pos = coords.copy()
            ref_com = ref_pos.mean(axis=0)

        H        = (coords - coords.mean(0)).T @ (ref_pos - ref_com)
        U, _, Vt = np.linalg.svd(H)
        d        = np.sign(np.linalg.det(Vt.T @ U.T))
        R        = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        coords   = (coords - coords.mean(0)) @ R.T + ref_com

        frames.append(coords.ravel())

    if len(frames) < 3:
        print("❌  Not enough valid frames for PCA.")
        return None

    positions = np.array(frames)                          # (n_frames, 3*n_atoms)
    n_comp    = min(n_components, min(positions.shape) - 1)
    centered  = (positions - positions.mean(axis=0)).astype(np.float64)

    print(f"   PCA: ({positions.shape[0]} frames, {positions.shape[1]} features) "
      f"from {positions.shape[1]//3} CA-atoms → {n_comp} PCs")

    pca_model = _PCA(n_components=n_comp, svd_solver="full")
    proj      = pca_model.fit_transform(centered)         # (n_frames, n_comp)

    evr    = pca_model.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    eigs   = pca_model.explained_variance_
    evecs  = pca_model.components_                        # (n_comp, 3*n_atoms)

    print(f"   PC1 = {evr[0]*100:.1f}%   PC2 = {evr[1]*100:.1f}%   "
          f"top-{n_comp} cumulative = {cumvar[-1]*100:.1f}%")

    # ── Residue cross-correlation matrix ─────────────────────────────────────
    n_atoms  = positions.shape[1] // 3
    evecs_3d = evecs.reshape(n_comp, n_atoms, 3)
    cov      = np.einsum('kia,kja,k->ij', evecs_3d, evecs_3d, np.abs(eigs))
    var      = np.diag(cov)
    denom    = np.sqrt(np.outer(var, var))
    cross_corr = np.where(denom > 0, cov / denom, 0.0).astype(np.float32)

    return dict(
        projections              = proj,
        explained_variance_ratio = evr,
        cumulative_variance      = cumvar,
        cross_correlation        = cross_corr,
        n_components             = n_comp,
    )


def compute_fel(pca_result, temperature=PCA_TEMP,
                n_bins=PCA_NBINS, sigma=PCA_SIGMA):
    """
    Boltzmann inversion of the PC1/PC2 density histogram to obtain a
    2-D Free-Energy Landscape.  F = –kBT ln(ρ),  shifted so min(F) = 0.
    """
    kBT  = 0.008314462 * temperature          # kJ mol⁻¹
    proj = pca_result["projections"]
    pc1, pc2 = proj[:, 0], proj[:, 1]

    pad_x = (pc1.max() - pc1.min()) * 0.20
    pad_y = (pc2.max() - pc2.min()) * 0.20
    rng   = [[pc1.min() - pad_x, pc1.max() + pad_x],
             [pc2.min() - pad_y, pc2.max() + pad_y]]

    hist, xe, ye = np.histogram2d(pc1, pc2, bins=n_bins,
                                  range=rng, density=True)
    hist_s = gaussian_filter(hist, sigma=sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(hist_s > 0, -kBT * np.log(hist_s), np.nan)
    F -= np.nanmin(F)

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])

    return dict(F=F, xcenters=xc, ycenters=yc,
                xedges=xe, yedges=ye,
                pc1=pc1, pc2=pc2, kBT=kBT, temperature=temperature)


# =============================================================================
# 📊  REPORT FIGURE  (2 × 2 layout)
# =============================================================================

def plot_pca_report(obj_name,
                    selection  = PCA_SEL,
                    n_components = PCA_NCOMP,
                    n_bins     = PCA_NBINS,
                    sigma      = PCA_SIGMA,
                    temperature = PCA_TEMP,
                    output     = OUTPUT_PNG):
    """
    Full PCA / FEL report saved to *output*.

    Panel layout:
      Top-left    – Free-Energy Landscape (PC1 vs PC2)
      Top-right   – Residue cross-correlation matrix
      Bottom-left – Explained-variance bar chart (first 10 PCs)
      Bottom-right – PC1 & PC2 projection histograms + KDE
    """
    from scipy.stats import gaussian_kde
    from matplotlib.gridspec import GridSpec

    print(f"\n💠  Essential Dynamics Analysis — '{obj_name}'")

    pca = compute_pca(obj_name, selection, n_components)
    if pca is None:
        return None

    fel = compute_fel(pca, temperature, n_bins, sigma)

    evr     = pca["explained_variance_ratio"]
    nc      = pca["n_components"]
    F       = fel["F"]
    xc, yc  = fel["xcenters"], fel["ycenters"]
    pc1, pc2 = fel["pc1"], fel["pc2"]

    # ── Figure skeleton ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Essential Dynamics  —  PCA Report",
                 fontsize=15, fontweight="bold")

    gs = GridSpec(2, 2, figure=fig,
                  hspace=0.32, wspace=0.35,
                  top=0.94, bottom=0.06, left=0.07, right=0.97)

    ax_fel = fig.add_subplot(gs[0, 0])
    ax_cc  = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_kde = fig.add_subplot(gs[1, 1])

    # ── Panel 1: Free-Energy Landscape ───────────────────────────────────────
    F_plot = np.where(np.isnan(F), np.nanmax(F), F)
    XX, YY = np.meshgrid(xc, yc)
    levels = np.linspace(0, np.nanpercentile(F, 97), 30)

    cf = ax_fel.contourf(XX, YY, F_plot.T, levels=levels,
                         cmap="RdYlBu_r", extend="max")
    ax_fel.contour(XX, YY, F_plot.T, levels=levels[::5],
                   colors="k", linewidths=0.4, alpha=0.5)

    cbar = fig.colorbar(cf, ax=ax_fel, fraction=0.046, pad=0.04)
    cbar.set_label("Free Energy (kJ mol⁻¹)", fontsize=10)
    F_max  = np.nanpercentile(F, 97)
    _step  = max(1, int(round(F_max / 6)))
    cbar.set_ticks(range(0, int(F_max) + _step, _step))

    ax_fel.plot(pc1, pc2, color="white", lw=0.25, alpha=0.3, rasterized=True)
    ax_fel.scatter(pc1[0],  pc2[0],  c="lime", s=130, marker="*",
                   zorder=5, edgecolors="k", lw=0.7, label="Start")
    ax_fel.scatter(pc1[-1], pc2[-1], c="red",  s=130, marker="*",
                   zorder=5, edgecolors="k", lw=0.7, label="End")

    ax_fel.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax_fel.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax_fel.set_title(f"Free-Energy Landscape  (T = {temperature:.0f} K)",
                     fontsize=12, fontweight="bold")
    ax_fel.set_xlim(fel["xedges"][0], fel["xedges"][-1])
    ax_fel.set_ylim(fel["yedges"][0], fel["yedges"][-1])
    ax_fel.legend(fontsize=10, loc="upper right", frameon=True)
    ax_fel.grid(True, color="white", alpha=0.15, linestyle="--", linewidth=0.5)

    # ── Panel 2: Cross-Correlation Matrix ────────────────────────────────────
    cc = pca["cross_correlation"]
    im = ax_cc.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1,
                      aspect="auto", origin="lower", interpolation="nearest")
    fig.colorbar(im, ax=ax_cc, fraction=0.046, pad=0.04,
                 label="Cross-correlation")
    ax_cc.set_xlabel("Residue index", fontsize=11, fontweight="bold")
    ax_cc.set_ylabel("Residue index", fontsize=11, fontweight="bold")
    ax_cc.set_title("Residue Cross-Correlation Matrix",
                    fontsize=12, fontweight="bold")

    # ── Panel 3: Explained-Variance Bar Chart ────────────────────────────────
    n_show  = min(nc, 10)
    x_ticks = range(1, n_show + 1)

    bars = ax_bar.bar(x_ticks, evr[:n_show] * 100,
                      color="steelblue", alpha=0.85,
                      edgecolor="navy", linewidth=0.6)

    for bar, pct in zip(bars, evr[:n_show] * 100):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2 = ax_bar.twinx()
    ax2.plot(x_ticks, np.cumsum(evr[:n_show]) * 100,
             "o--", color="coral", lw=1.8, ms=5, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=10, color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax2.set_ylim(0, 105)
    ax2.axhline(80, ls=":", color="gray", alpha=0.6, lw=1.0)
    ax2.axhline(90, ls=":", color="gray", alpha=0.6, lw=1.0)
    ax2.legend(loc="center right", fontsize=9)

    ax_bar.set_xlabel("Principal Component", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("Explained Variance (%)", fontsize=11, fontweight="bold")
    ax_bar.set_title(f"First {n_show} PCs — Explained Variance",
                     fontsize=12, fontweight="bold")
    ax_bar.set_xticks(list(x_ticks))
    ax_bar.set_ylim(0, 105)
    ax_bar.grid(True, axis="y", color="skyblue", alpha=0.4, linestyle="--")
    ax_bar.set_axisbelow(True)

    # ── Panel 4: PC1 & PC2 Projection Histograms + KDE ───────────────────────
    for comp, label, color, idx in [(pc1, "PC1", "teal", 0),
                                    (pc2, "PC2", "darkorange", 1)]:
        pct = evr[idx] * 100
        ax_kde.hist(comp, bins=60, color=color, alpha=0.45,
                    edgecolor="k", linewidth=0.3, density=True, label=f"{label} ({pct:.1f}%)")
        xr = np.linspace(comp.min(), comp.max(), 300)
        ax_kde.plot(xr, gaussian_kde(comp)(xr), color=color, lw=2.0)
        ax_kde.axvline(comp.mean(), color=color, ls="--", lw=1.2)

    ax_kde.set_xlabel("Projection value", fontsize=11, fontweight="bold")
    ax_kde.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax_kde.set_title("PC1 & PC2 Projection Distributions",
                     fontsize=12, fontweight="bold")
    ax_kde.legend(fontsize=9)
    ax_kde.grid(True, color="lightgray", alpha=0.5, linestyle="--")

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"👑  PCA report saved → {output}")
    plt.close("all")
    time.sleep(0.3)
    os.system(f"open {output}")

    return output


# =============================================================================
# 🚀  MAIN PIPELINE  — runs automatically on drag-and-drop in PyMOL
# =============================================================================

def main():
    # start benchmark timer
    start_all = time.time() # start benchamrk timer
    traj = next(
        (f for ext in ("*.nc", "*.xtc", "*.trr", "*.dcd")
         for f in glob.glob(ext)),
        None
    )

    all_objects = cmd.get_names("objects")
    if not all_objects:
        print("❌  No objects loaded in PyMOL. Load topology + trajectory first.")
        return

    target = all_objects[0]
    print(f"✨  Target object: {target}")

    if traj:
        print(f"💫  Loading trajectory: {traj}")
        cmd.load_traj(traj, target, interval=MD_INTERVAL)
    else:
        print("ℹ️   No trajectory file found — using states already in PyMOL.")

    plot_pca_report(target)

    # Calculate total time
    total_elapsed = time.time() - start_all
    hours, rem = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print()  # <--- Prints an empty line
    print(f"🕰️ Total Execution Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# run the workflow in PyMOL
main()
