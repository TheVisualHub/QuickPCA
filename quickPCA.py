# quickPCA.py (ver 2.00)
# Essential Dynamics Analysis for MD trajectories.
# Author: Gleb Novikov
# © The Visual Hub 2026
# For educational use only.
# If you use QuickPCA in your research, please cite this tool.
# Any contributions toward its development are also appreciated.
# Last update 13/06/2026: the workflow now works in two modes:
# ── 1 -> PyMOL mode (drag-and-drop) ───────────────────────────────────────────────
#   Drag this script onto the PyMOL window.
#   Supported trajectory formats: .nc  .xtc  .trr  .dcd  (all via PyMOL)
#   Settings are read from the USER SETTINGS block below.
#
# ── 2 -> Terminal / standalone mode ───────────────────────────────────────────────
#   python quickPCA.py topology.pdb trajectory.nc
#   python quickPCA.py topology.pdb trajectory.dcd
#   python quickPCA.py topology.pdb               # multi-model PDB
#   python quickPCA.py topology.pdb traj.nc --interval 10 --temp 310
#   Supported trajectory formats: .nc  .dcd  (pure numpy / scipy)
#
# Output: PCA_Report.png
#
# Dependencies: numpy, scikit-learn, scipy, matplotlib
#   pip install numpy scikit-learn scipy matplotlib

import os
import sys
import glob
import time
import struct
import platform # added in ver 2.00
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# =============================================================================
# ⚙️  USER SETTINGS  — edit these before running
#     In terminal mode these are also overridable via CLI flags.
# =============================================================================

PCA_SEL     = "polymer and name CA"   # PyMOL selection  /  atom-name filter
PCA_NCOMP   = 10                      # number of PCs to compute (≥ 2)
PCA_NBINS   = 50                      # FEL histogram bins per axis
PCA_SIGMA   = 1.0                     # Gaussian σ (bin units) for FEL smoothing
PCA_TEMP    = 300.0                   # temperature (K) for Boltzmann inversion
MD_INTERVAL = 5                       # stride: take every Nth frame
OUTPUT_PNG  = "PCA_Report.png"

# =============================================================================
# 🔀  RUNTIME DETECTION
# =============================================================================

def _detect_pymol():
    """
    True only when executing INSIDE a live PyMOL process.

    When PyMOL drag-and-drops a script, sys.argv[0] is the PyMOL
    executable (contains 'pymol'), NOT the script filename.
    In plain  python quickPCA.py  sys.argv[0] is the script itself.
    This is the reliable discriminator — import success alone is not
    enough because pymol may be installed in the same environment.
    """
    argv0 = os.path.basename(sys.argv[0]).lower()
    if "pymol" not in argv0:
        return False
    try:
        from pymol import cmd as _cmd
        _cmd.get_version()
        return True
    except Exception:
        return False

PYMOL_MODE = _detect_pymol()

# =============================================================================
# 📂  PATH A — PyMOL coordinate extraction
#     Returns list of (n_CA, 3) float64 arrays, identical contract to Path B.
# =============================================================================

def _load_frames_pymol(interval=MD_INTERVAL, selection=PCA_SEL):
    """
    Use PyMOL cmd to load a trajectory (all supported formats) and extract
    per-state Cα coordinates.  Returns (frames, obj_name).
    """
    from pymol import cmd

    traj = next(
        (f for ext in ("*.nc", "*.xtc", "*.trr", "*.dcd")
         for f in glob.glob(ext)),
        None
    )

    all_objects = cmd.get_names("objects")
    if not all_objects:
        print("❌  No objects loaded in PyMOL. Load topology first.")
        return None, None

    obj = all_objects[0]
    print(f"✨  Target object : {obj}")

    if traj:
        print(f"💫  Loading trajectory: {traj}")
        cmd.load_traj(traj, obj, interval=interval)
    else:
        print("ℹ️   No trajectory found — using states already in PyMOL.")

    n_states = cmd.count_states(obj)
    if n_states < 3:
        print(f"❌  Need ≥ 3 frames, found {n_states}.")
        return None, obj

    print(f"   Extracting '{selection}' from {n_states} states …")

    ref_pos = ref_com = None
    frames  = []

    for state in range(1, n_states + 1):
        model  = cmd.get_model(f"({obj}) and ({selection})", state=state)
        coords = np.array([a.coord for a in model.atom], dtype=np.float64)
        if coords.size == 0:
            print(f"   ⚠️  State {state}: no atoms matched — skipping.")
            continue

        if ref_pos is None:
            ref_pos = coords.copy()
            ref_com = ref_pos.mean(axis=0)

        frames.append(coords)

    print(f"   {len(frames)} frames extracted.")
    return frames, obj


# =============================================================================
# 📂  PATH B — pure numpy / scipy coordinate extraction  (terminal mode)
# =============================================================================

def _parse_pdb_topology(pdb_path, atom_name="CA"):
    """
    Read a single-model PDB; return (ca_indices, ref_coords).
    atom_name is the bare name field (e.g. 'CA'), NOT a PyMOL selection.
    """
    all_coords, ca_indices = [], []
    atom_counter = 0

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            name = line[12:16].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                atom_counter += 1
                continue
            all_coords.append([x, y, z])
            if name == atom_name:
                ca_indices.append(atom_counter)
            atom_counter += 1

    if not ca_indices:
        raise ValueError(f"No atoms named '{atom_name}' found in {pdb_path}")

    all_coords = np.array(all_coords, dtype=np.float64)
    ca_indices  = np.array(ca_indices, dtype=np.int64)
    print(f"   Topology: {len(all_coords)} atoms total, "
          f"{len(ca_indices)} CA atoms selected.")
    return ca_indices, all_coords[ca_indices]


def _parse_pdb_multimodel(pdb_path, interval=MD_INTERVAL):
    """Extract CA coords from every MODEL block (or single model) of a PDB."""
    frames, current = [], []
    in_model = has_model = False

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "MODEL":
                has_model = in_model = True
                current = []
            elif rec == "ENDMDL":
                if current:
                    frames.append(np.array(current, dtype=np.float64))
                in_model = False
            elif rec in ("ATOM", "HETATM") and line[12:16].strip() == "CA":
                try:
                    xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                except ValueError:
                    continue
                if has_model and in_model:
                    current.append(xyz)
                elif not has_model:
                    current.append(xyz)

    if not has_model and current:
        frames.append(np.array(current, dtype=np.float64))

    if not frames:
        raise ValueError(f"No CA coordinates found in {pdb_path}")

    frames = frames[::interval]
    print(f"   Multi-model PDB: {len(frames)} frames after stride={interval}.")
    return frames


def _read_dcd(dcd_path, ca_indices, interval=MD_INTERVAL):
    """
    Parse CHARMM/NAMD DCD (Fortran-record binary).
    Records: [header][title][natom][X][Y][Z] × n_frames
    """
    frames = []

    with open(dcd_path, "rb") as fh:

        def read_record():
            raw = fh.read(4)
            if len(raw) < 4:
                return None
            n = struct.unpack("i", raw)[0]
            data = fh.read(n)
            fh.read(4)
            return data

        hdr = read_record()
        if hdr is None or hdr[:4] not in (b"CORD", b"VELD"):
            raise ValueError("Not a valid DCD file (bad magic).")
        n_frames_hdr = struct.unpack("i", hdr[4:8])[0]

        read_record()                               # title — discard
        n_atoms = struct.unpack("i", read_record())[0]
        print(f"   DCD: {n_frames_hdr} frames declared, {n_atoms} atoms/frame.")

        if ca_indices.max() >= n_atoms:
            raise ValueError(f"CA index {ca_indices.max()} ≥ n_atoms {n_atoms}.")

        frame_idx = 0
        while True:
            xr = read_record()
            if xr is None:
                break
            yr = read_record()
            zr = read_record()
            if yr is None or zr is None:
                break

            if frame_idx % interval == 0:
                x = np.frombuffer(xr, dtype=np.float32).astype(np.float64)
                y = np.frombuffer(yr, dtype=np.float32).astype(np.float64)
                z = np.frombuffer(zr, dtype=np.float32).astype(np.float64)
                # some writers prepend a unit-cell record; trim to n_atoms
                if len(x) > n_atoms:
                    x, y, z = x[-n_atoms:], y[-n_atoms:], z[-n_atoms:]
                frames.append(np.stack([x[ca_indices],
                                        y[ca_indices],
                                        z[ca_indices]], axis=1))
            frame_idx += 1

    print(f"   DCD: {len(frames)} frames loaded after stride={interval}.")
    return frames


def _read_nc(nc_path, ca_indices, interval=MD_INTERVAL):
    """AMBER NetCDF via scipy.io.netcdf_file — variable: coordinates[n,atoms,3]."""
    from scipy.io import netcdf_file

    with netcdf_file(nc_path, "r", mmap=False) as nc:
        if "coordinates" not in nc.variables:
            raise ValueError("NetCDF file has no 'coordinates' variable.")
        coords_all = nc.variables["coordinates"].data   # (n_frames, n_atoms, 3)
        n_frames, n_atoms, _ = coords_all.shape
        print(f"   NC: {n_frames} frames, {n_atoms} atoms/frame.")
        if ca_indices.max() >= n_atoms:
            raise ValueError(f"CA index {ca_indices.max()} ≥ n_atoms {n_atoms}.")
        frames = [coords_all[i][ca_indices].astype(np.float64)
                  for i in range(0, n_frames, interval)]

    print(f"   NC: {len(frames)} frames loaded after stride={interval}.")
    return frames


def _load_frames_numpy(pdb_path, traj_path=None, interval=MD_INTERVAL):
    """
    Terminal-mode loader.  Returns list of (n_CA, 3) float64 arrays.
    """
    ca_indices, _ = _parse_pdb_topology(pdb_path)

    if traj_path is None:
        return _parse_pdb_multimodel(pdb_path, interval)

    ext = os.path.splitext(traj_path)[1].lower()
    if ext == ".dcd":
        return _read_dcd(traj_path, ca_indices, interval)
    elif ext == ".nc":
        return _read_nc(traj_path, ca_indices, interval)
    elif ext in (".xtc", ".trr"):
        print(f"❌  Format '{ext}' is not supported in terminal mode.\n"
              f"   Install MDAnalysis and use the PyMOL version, or convert to .dcd/.nc")
        sys.exit(1)
    else:
        raise ValueError(f"Unsupported trajectory format: {ext}")


# =============================================================================
# 🔬  CORE MATH  (shared by both modes — input is always list of (n_CA,3) arrays)
# =============================================================================

def compute_pca(frames, n_components=PCA_NCOMP):
    """
    Kabsch SVD alignment → mean-centre → sklearn PCA.
    *frames* — list of (n_CA, 3) float64 arrays.
    Returns result dict or None on failure.
    """
    try:
        from sklearn.decomposition import PCA as _PCA
    except ImportError:
        print("❌  scikit-learn not found.  pip install scikit-learn")
        return None

    if len(frames) < 3:
        print(f"❌  Need ≥ 3 frames, found {len(frames)}.")
        return None

    print(f"   PCA input: {len(frames)} frames, {frames[0].shape[0]} Cα atoms.")

    ref_pos = frames[0].copy()
    ref_com = ref_pos.mean(axis=0)
    flat    = []

    for coords in frames:
        H        = (coords - coords.mean(0)).T @ (ref_pos - ref_com)
        U, _, Vt = np.linalg.svd(H)
        d        = np.sign(np.linalg.det(Vt.T @ U.T))
        R        = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        aligned  = (coords - coords.mean(0)) @ R.T + ref_com
        flat.append(aligned.ravel())

    positions = np.array(flat, dtype=np.float64)          # (n_frames, 3*n_CA)
    n_comp    = min(n_components, min(positions.shape) - 1)
    centered  = positions - positions.mean(axis=0)

    print(f"   PCA: ({positions.shape[0]} frames × {positions.shape[1]} features) "
          f"→ {n_comp} PCs")

    pca_model = _PCA(n_components=n_comp, svd_solver="full")
    proj      = pca_model.fit_transform(centered)

    evr    = pca_model.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    eigs   = pca_model.explained_variance_
    evecs  = pca_model.components_                        # (n_comp, 3*n_CA)

    print(f"   PC1 = {evr[0]*100:.1f}%   PC2 = {evr[1]*100:.1f}%   "
          f"top-{n_comp} cumulative = {cumvar[-1]*100:.1f}%")

    n_atoms   = positions.shape[1] // 3
    evecs_3d  = evecs.reshape(n_comp, n_atoms, 3)
    cov       = np.einsum('kia,kja,k->ij', evecs_3d, evecs_3d, np.abs(eigs))
    var       = np.diag(cov)
    denom     = np.sqrt(np.outer(var, var))
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
    """Boltzmann inversion of PC1/PC2 density → 2-D FEL.  min(F) = 0."""
    kBT  = 0.008314462 * temperature
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
    return dict(F=F, xcenters=xc, ycenters=yc, xedges=xe, yedges=ye,
                pc1=pc1, pc2=pc2, kBT=kBT, temperature=temperature)


# =============================================================================
# 📊  REPORT FIGURE  (2 × 2 — shared by both modes)
# =============================================================================

def plot_pca_report(pca, fel, output=OUTPUT_PNG, max_pcs=10):
    """
    Panel layout:
      Top-left     – Free-Energy Landscape (PC1 vs PC2)
      Top-right    – Residue cross-correlation matrix
      Bottom-left  – Explained-variance bar chart (first 10 PCs)
      Bottom-right – PC1 & PC2 projection histograms + KDE
    """
    from scipy.stats import gaussian_kde
    from matplotlib.gridspec import GridSpec

    evr      = pca["explained_variance_ratio"]
    nc       = pca["n_components"]
    F        = fel["F"]
    xc, yc   = fel["xcenters"], fel["ycenters"]
    pc1, pc2 = fel["pc1"], fel["pc2"]

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

    # ── FEL ──────────────────────────────────────────────────────────────────
    F_plot = np.where(np.isnan(F), np.nanmax(F), F)
    XX, YY = np.meshgrid(xc, yc)
    levels  = np.linspace(0, np.nanpercentile(F, 97), 30)
    cf = ax_fel.contourf(XX, YY, F_plot.T, levels=levels,
                         cmap="RdYlBu_r", extend="max")
    ax_fel.contour(XX, YY, F_plot.T, levels=levels[::5],
                   colors="k", linewidths=0.4, alpha=0.5)
    cbar = fig.colorbar(cf, ax=ax_fel, fraction=0.046, pad=0.04)
    cbar.set_label("Free Energy (kJ mol⁻¹)", fontsize=10)
    F_max = np.nanpercentile(F, 97)
    _step = max(1, int(round(F_max / 6)))
    cbar.set_ticks(range(0, int(F_max) + _step, _step))
    ax_fel.plot(pc1, pc2, color="white", lw=0.25, alpha=0.3, rasterized=True)
    ax_fel.scatter(pc1[0],  pc2[0],  c="lime", s=130, marker="*",
                   zorder=5, edgecolors="k", lw=0.7, label="Start")
    ax_fel.scatter(pc1[-1], pc2[-1], c="red",  s=130, marker="*",
                   zorder=5, edgecolors="k", lw=0.7, label="End")
    ax_fel.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax_fel.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax_fel.set_title(f"Free-Energy Landscape  (T = {fel['temperature']:.0f} K)",
                     fontsize=12, fontweight="bold")
    ax_fel.set_xlim(fel["xedges"][0], fel["xedges"][-1])
    ax_fel.set_ylim(fel["yedges"][0], fel["yedges"][-1])
    ax_fel.legend(fontsize=10, loc="upper right", frameon=True)
    ax_fel.grid(True, color="white", alpha=0.15, linestyle="--", linewidth=0.5)

    # ── Cross-correlation ─────────────────────────────────────────────────────
    cc = pca["cross_correlation"]
    im = ax_cc.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1,
                      aspect="auto", origin="lower", interpolation="nearest")
    fig.colorbar(im, ax=ax_cc, fraction=0.046, pad=0.04,
                 label="Cross-correlation")
    ax_cc.set_xlabel("Residue index", fontsize=11, fontweight="bold")
    ax_cc.set_ylabel("Residue index", fontsize=11, fontweight="bold")
    ax_cc.set_title("Residue Cross-Correlation Matrix",
                    fontsize=12, fontweight="bold")

    # ── Explained variance ────────────────────────────────────────────────────
    n_show = min(nc, max_pcs)
    x_ticks = range(1, n_show + 1)
    bars = ax_bar.bar(x_ticks, evr[:n_show] * 100,
                      color="steelblue", alpha=0.85,
                      edgecolor="navy", linewidth=0.6)
    label_fs = max(5, 8 - max(0, n_show - 10))
    for bar, pct in zip(bars, evr[:n_show] * 100):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=label_fs, fontweight="bold")
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
    ax_bar.tick_params(axis='x', labelsize=label_fs)
    ax_bar.set_ylim(0, 105)
    ax_bar.grid(True, axis="y", color="skyblue", alpha=0.4, linestyle="--")
    ax_bar.set_axisbelow(True)

    # ── PC1 / PC2 distributions ───────────────────────────────────────────────
    for comp, label, color, idx in [(pc1, "PC1", "teal", 0),
                                    (pc2, "PC2", "darkorange", 1)]:
        pct = evr[idx] * 100
        ax_kde.hist(comp, bins=60, color=color, alpha=0.45,
                    edgecolor="k", linewidth=0.3, density=True,
                    label=f"{label} ({pct:.1f}%)")
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
    # ── Open Results ─────────────────────────────────────────────────────────
    if platform.system() == "Darwin":
        os.system(f"open {output}")
    elif platform.system() == "Windows":
        os.system(f"start {output}")
    else:
        os.system(f"xdg-open {output}")


# =============================================================================
# 🚀  MAIN — two entry points, one shared pipeline
# =============================================================================

def main():
    start_all = time.time()

    ncomp  = PCA_NCOMP
    temp   = PCA_TEMP
    output = OUTPUT_PNG

    if PYMOL_MODE:
        # ── PyMOL drag-and-drop ───────────────────────────────────────────────
        print(f"\n💠  Essential Dynamics Analysis  [PyMOL mode]")
        print(f"   Selection : {PCA_SEL}")
        print(f"   Stride    : every {MD_INTERVAL} frames")
        print(f"   Temp      : {temp} K\n")

        frames, obj = _load_frames_pymol(interval=MD_INTERVAL, selection=PCA_SEL)
        if frames is None:
            return

    else:
        # ── Terminal / standalone ─────────────────────────────────────────────
        import argparse

        parser = argparse.ArgumentParser(
            description="Essential Dynamics PCA — terminal mode (no PyMOL).",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python quickPCA.py protein.pdb traj.nc
  python quickPCA.py protein.pdb traj.dcd
  python quickPCA.py multimodel.pdb
  python quickPCA.py protein.pdb traj.nc --interval 10 --temp 310
            """
        )
        parser.add_argument("pdb",  help="Topology PDB file (single model)")
        parser.add_argument("traj", nargs="?", default=None,
                            help="Trajectory .nc or .dcd  "
                                 "(auto-discovered if omitted)")
        parser.add_argument("--interval", type=int,   default=MD_INTERVAL)
        parser.add_argument("--temp",     type=float, default=PCA_TEMP)
        parser.add_argument("--ncomp",    type=int,   default=PCA_NCOMP)
        parser.add_argument("--output",   default=OUTPUT_PNG)
        args = parser.parse_args()

        ncomp  = args.ncomp
        temp   = args.temp
        output = args.output

        traj = args.traj
        if traj is None:
            traj = next(
                (f for ext in ("*.nc", "*.dcd") for f in glob.glob(ext)),
                None
            )
            if traj:
                print(f"💫  Auto-discovered trajectory: {traj}")

        print(f"\n💠  Essential Dynamics Analysis  [terminal mode]")
        print(f"   Topology  : {args.pdb}")
        print(f"   Trajectory: {traj or '(none — multi-model PDB)'}")
        print(f"   Stride    : every {args.interval} frames")
        print(f"   Temp      : {temp} K\n")

        frames = _load_frames_numpy(args.pdb, traj, interval=args.interval)

    # ── Shared pipeline ───────────────────────────────────────────────────────
    if len(frames) < 3:
        print(f"❌  Only {len(frames)} frames — need at least 3.")
        return

    pca = compute_pca(frames, n_components=ncomp)
    if pca is None:
        return

    fel = compute_fel(pca, temperature=temp)
    plot_pca_report(pca, fel, output=output, max_pcs=ncomp)

    total = time.time() - start_all
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    print()
    print(f"🕰️  Total Execution Time: {int(h)}h {int(m)}m {int(s)}s")


# PyMOL executes via exec() so __name__ != "__main__" — call main() directly.
main()