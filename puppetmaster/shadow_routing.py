"""Pure construction of opt-in, non-interfering shadow-routing evidence."""
from __future__ import annotations

from typing import Any


def shadow_evidence(
    *,
    production_model_id: str,
    counterfactual_model_id: str,
    policy: str,
) -> dict[str, Any]:
    """Describe a counterfactual without authorizing it for dispatch."""
    return {
        "enabled": True,
        "policy": policy,
        "production_model_id": production_model_id,
        "counterfactual_model_id": counterfactual_model_id,
        "production_selection_changed": False,
    }
