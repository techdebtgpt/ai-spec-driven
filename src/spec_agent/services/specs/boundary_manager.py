from __future__ import annotations

from typing import List
from uuid import uuid4

from ...domain.models import BoundarySpec, Plan


class BoundaryManager:
    """
    Detects cross-boundary changes and drafts human + machine-readable specs.
    """

    def required_specs(self, plan: Plan) -> List[BoundarySpec]:
        boundary_specs: List[BoundarySpec] = []
        for step in plan.steps:
            if "boundary" in (step.notes or "").lower():
                boundary_specs.append(
                    BoundarySpec(
                        id=str(uuid4()),
                        task_id=plan.task_id,
                        boundary_name="AutoDetectedBoundary",
                        human_description="Defines responsibilities between Context Engine and Patch Engine.",
                        diagram_text="""sequenceDiagram
    participant Engineer
    participant Context
    participant Patch
    Engineer->>Context: request files + analyses
    Context-->>Engineer: structured context bundle
    Engineer->>Patch: approve stepwise change
""",
                        machine_spec={
                            "name": "auto-detected-boundary",
                            "actors": ["ContextEngine", "PatchEngine"],
                            "interfaces": ["context_bundle", "patch_request"],
                            "invariants": [
                                "Patch requests must reference a previously approved plan step."
                            ],
                        },
                    )
                )
        return boundary_specs


