from spec_agent.domain.models import PlanStep


def test_plan_step_serialization_round_trip() -> None:
    step = PlanStep(
        description="Touch auth pipeline",
        target_files=["src/auth.py", "src/login.py"],
        notes="Ensure backward compatibility",
    )

    restored = PlanStep.from_dict(step.to_dict())

    assert restored == step
