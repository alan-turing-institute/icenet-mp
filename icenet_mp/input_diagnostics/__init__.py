"""Input diagnostics — unsupervised analysis of input variable structure.

Provides VIF (multicollinearity), PCA (variance-based importance), and EOF
(spatiotemporal mode decomposition) for understanding the structure of model
input variables before training or feature selection.

These methods analyse input variables alone — they do not use the target
(SIC) variable, so they measure information content and redundancy in the
feature space, not predictive power.

"""

from .data import build_datasets, build_sample_matrix, resolve_datasets
from .eof import compute_eof, run_eof_analysis
from .pca import compute_pca, run_pca_analysis
from .vif import compute_vif, run_vif_analysis

__all__ = [
    "build_datasets",
    "build_sample_matrix",
    "compute_eof",
    "compute_pca",
    "compute_vif",
    "resolve_datasets",
    "run_eof_analysis",
    "run_pca_analysis",
    "run_vif_analysis",
]
