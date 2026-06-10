# QuickPCA

*A streamlined workflow for Essential Dynamics Analysis of Molecular Dynamics trajectories.*

QuickPCA is a lightweight Python script that performs Principal Component Analysis (PCA) on molecular dynamics trajectories in PyMOL. It automatically loads trajectory data from the curent directory and generates a publication-ready report.

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
