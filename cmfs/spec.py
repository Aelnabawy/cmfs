from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

@dataclass
class CausalSpec:
    """Causal structure the selection works within.

    exposure     binary column (1 = disease, 0 = control)
    outcome      continuous functional outcome
    confounders  pre-exposure common causes (adjusted on every path)
    environment  optional study-phase / site variable (adjusted and used for invariance checks)
    tiers        ordered mediator tiers, e.g. {"blood": [...], "neuro": [...]}; earlier tiers may affect
                 later tiers and the outcome directly; the last tier affects the outcome
    strata       optional column splitting the exposed group (e.g. lesion hemisphere 'L'/'R');
                 selection is then run per stratum against all controls
    family       optional function mapping a feature name to its measurement family (one per family)
    """
    exposure: str
    outcome: str
    confounders: List[str]
    tiers: Dict[str, List[str]]
    environment: Optional[str] = None
    strata: Optional[str] = None
    family: Optional[Callable[[str], str]] = None

    def tier_names(self): return list(self.tiers.keys())
    def tier_of(self):
        return {f: t for t, cols in self.tiers.items() for f in cols}
