from pathlib import Path

from titan.spec import AgentSpec


def test_specs_valid():
    specs_dir = Path("specs")
    new_specs = [
        "scope.titan.yaml",
        "logic.titan.yaml",
        "mythos.titan.yaml",
        "bridge.titan.yaml",
        "meta.titan.yaml",
        "pattern.titan.yaml",
    ]

    for spec_name in new_specs:
        spec_path = specs_dir / spec_name
        assert spec_path.exists(), f"{spec_name} not found"

        # This will raise SpecValidationError if invalid
        spec = AgentSpec.from_file(spec_path)
        assert spec.name in ["scope", "logic", "mythos", "bridge", "meta", "pattern"]
        print(f"Spec {spec_name} is valid")


if __name__ == "__main__":
    test_specs_valid()
    print("All specs valid")
