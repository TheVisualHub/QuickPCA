# 💠 QuickPCA: welcome to the universe of eigenvectors!

<img 
    src="https://github.com/TheVisualHub/VisualFactory/blob/main/assets/quickPCA_logo.png" 
    alt="QuickPCA Logo" 
    width="800">

QuickPCA is a lightweight Python tool for Essential Dynamics Analysis of molecular dynamics trajectories in PyMOL. It automatically detects and loads MD trajectories, performs Principal Component Analysis, and generates a publication-ready report featuring free-energy landscapes, residue cross-correlation maps, explained variance profiles and principal component projections.

Molecular dynamics trajectories have a massive amount of data. If your protein has 1,000 atoms, each frame has 3,000 coordinates (X, Y, Z). If you have 10,000 frames, that is 30 million data points! PCA analyzes this massive dataset and reduces the dimensions. It figures out which atomic movements are just random "noise" and which are the "signals" capturing functional-relevant motions. This algorithm compresses thousands of dimensions down into just two (PC1 and PC2) while preserving the most important information.

Unlike common PCA approaches, which construct and diagonalize the covariance matrix, quickPCA performs SVD decomposition using scikit-learn directly on the (n_frames × 3N) data matrix. This avoids the costly step of diagonalizing the covariance matrix, making the approach faster and more numerically stable, while producing identical principal components. The residue cross-correlation matrix is subsequently recovered analytically from the PCA eigenvectors and eigenvalues, without revisiting the raw trajectory data.

## 👤 Author

This code was developed by **Gleb Novikov**

## 🔭 Features (ver. 2.00)

- Script runs in PyMol or in standalone mode
- Cross-platform support: Windows, Linux, MacOS
- Structural alignment using the Kabsch algorithm
- Principal Component Analysis (Essential Dynamics)
- Free-Energy Landscape (PC1 vs PC2)
- Residue cross-correlation matrix
- Explained variance analysis
- PC projection distributions
- Automated report generation

## 🛠️ Requirements

- PyMOL
- NumPy
- SciPy
- scikit-learn
- Matplotlib

```bash
pip install numpy scipy scikit-learn matplotlib
```

## ⚜️ Usage
For PyMol mode:
1. Place `quickPCA.py` in the same directory as your PDB and trajectory files.
2. Open your structure in PyMOL.
3. Drag and drop `quickPCA.py` into the PyMOL window.
   
For standalone mode:
1. Place `quickPCA.py` in the same directory or whatever you want.
2. Use the following commands in the terminal:

```bash
# With a topology.pdb + netcdf or dcd trajectory
python quickPCA.py topology.pdb trajectory.nc

# Multi-model PDB (no separate trajectory)
python quickPCA.py topology.pdb

# With optional arguments
python quickPCA.py topology.pdb trajectory.nc --interval 10 --temp 310 --ncomp 20
```

Supported trajectory formats:

```
.xtc  .trr (only with PyMol)
.dcd  .nc (for both PyMol and standalone runs)
```

## ⚙️ Output

QuickPCA generates:

- `PCA_Report.png`

The report includes:

- Free-Energy Landscape
- Residue Cross-Correlation Matrix
- Explained Variance Plot
- PC1/PC2 Projection Distributions

## ⚖️ License

Released under the MIT License. If QuickPCA is used in any capacity that contributes to results presented in a publication, thesis, report, or any other form of scholarly or professional work, appropriate citation of QuickPCA is strongly encouraged.
