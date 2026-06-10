# 💠 Welcome to the universe of eigenvectors!

<img 
    src="https://github.com/TheVisualHub/VisualFactory/blob/main/assets/quickPCA_logo.png" 
    alt="QuickPCA Logo" 
    width="800">

QuickPCA is a lightweight Python tool for Essential Dynamics Analysis of molecular dynamics trajectories in PyMOL. It automatically detects and loads MD trajectories, performs Principal Component Analysis, and generates a publication-ready report featuring free-energy landscapes, residue cross-correlation maps, explained variance profiles and principal component projections.

Molecular dynamics trajectories have a massive amount of data. If your protein has 1,000 atoms, each frame has 3,000 coordinates (X, Y, Z). If you have 10,000 frames, that is 30 million data points! PCA analyzes this massive dataset and reduces the dimensions. It figures out which atomic movements are just random "noise" and which are the "signals" capturing functional-relevant motions. This algorithm compresses thousands of dimensions down into just two (PC1 and PC2) while preserving the most important information.

## 👤 Author

This code was developed by **Gleb Novikov**

## 🔭 Features

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

1. Place `quickPCA.py` in the same directory as your PDB and trajectory files.
2. Open your structure in PyMOL.
3. Drag and drop `quickPCA.py` into the PyMOL window.

Supported trajectory formats:

```
.xtc  .trr  .dcd  .nc
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
