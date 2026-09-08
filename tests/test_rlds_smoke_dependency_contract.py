"""69b dependency installation regression tests (no network or GPU)."""
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rlds_smoke_recovery", ROOT / "tools/colab/rlds_smoke_recovery.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_promise_is_only_source_build_exception(tmp_path):
    calls = []

    def fake_exec(cmd, label, log_dir, **kwargs):
        calls.append((list(cmd), label))

    module.install_conversion_deps(
        "uv", tmp_path / "venv/bin/python", tmp_path / "logs",
        exec_command=fake_exec,
    )
    assert len(calls) == 2

    build, label = calls[0]
    assert label == "rlds-promise-build"
    assert "--no-deps" in build
    assert build[build.index("--no-binary") + 1] == "promise"
    assert "--no-binary-package" not in build
    assert "promise==2.3" in build

    install, label = calls[1]
    assert label == "rlds-conversion-deps"
    assert install[install.index("--only-binary") + 1] == ":all:"
    assert "tensorflow-cpu==2.17.1" in install
    assert "tensorflow-datasets==4.9.6" in install
    assert "tensorflow-metadata==1.16.1" in install
    assert "googleapis-common-protos==1.65.0" in install
    assert "protobuf==3.20.3" in install
    assert "av==12.3.0" in install
    assert "promise==2.3" not in install


def test_verification_contract_pins_metadata_stack():
    assert "tensorflow-metadata':'1.16.1" in module.CHECK
    assert "googleapis-common-protos':'1.65.0" in module.CHECK
    assert "protobuf':'3.20.3" in module.CHECK
    assert "tensorflow-datasets':'4.9.6" in module.CHECK
    assert "tensorflow-cpu':'2.17.1" in module.CHECK


def test_dependency_failure_stops_before_conversion(tmp_path):
    calls = []

    def fake_exec(cmd, label, log_dir, **kwargs):
        calls.append(label)
        if label == "rlds-promise-build":
            raise RuntimeError("build failed")

    try:
        module.install_conversion_deps(
            "uv", tmp_path / "venv/bin/python", tmp_path / "logs",
            exec_command=fake_exec,
        )
    except RuntimeError as exc:
        assert str(exc) == "build failed"
    else:
        raise AssertionError("Dependency failure was not propagated")
    assert calls == ["rlds-promise-build"]
