"""Input explainability — supervised analysis of which inputs drive model predictions.

Provides methods for understanding how input variables contribute to predicting a
specific target (e.g., next-day sea-ice concentration). Unlike diagnostics, these
methods are **supervised**: they measure predictive contribution, not just information
content or redundancy.

**Available methods:**
- ``rf`` — Random Forest with permutation importance and interaction analysis.

**Extensibility:** New explainability methods (SHAP, partial dependence, etc.) can be
added by implementing the :class:`ExplainabilityMethod` protocol in a new submodule.

"""

from .base import ExplainabilityMethod, ExplainabilityResult
from .rf import compute_rf_importance, run_rf_analysis

__all__ = [
    "ExplainabilityMethod",
    "ExplainabilityResult",
    "compute_rf_importance",
    "run_rf_analysis",
]
