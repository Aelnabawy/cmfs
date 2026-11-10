"""cmfs - Causal Mediation Feature Selection.

Selects diagnostic biomarkers by their route-specific mediated effect on a functional
outcome within a tiered causal graph, assigns evidence-graded causal roles, and uses
cross-fitted transformations so learned mediators do not inflate causal estimates.
"""
from .spec import CausalSpec
from .pipeline import CMFS
__all__ = ["CausalSpec", "CMFS"]
__version__ = "0.1.0"
