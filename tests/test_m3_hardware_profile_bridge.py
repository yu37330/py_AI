import pytest

from tools.benchmark import m3_batch_probe_common
from tools.benchmark import m3_hardware_guard


def _fake_nvidia_smi(monkeypatch, *, name: str, memory_mib: int):
    def fake_check_output(args, text=True):
        query = " ".join(args)
        if "--query-gpu=name" in query:
            return name + "\n"
        if "--query-gpu=memory.total" in query:
            return str(memory_mib) + "\n"
        raise AssertionError(args)

    monkeypatch.setattr(m3_batch_probe_common.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(m3_hardware_guard.subprocess, "check_output", fake_check_output)


def test_historical_batch_probe_gpu_info_remains_a100_only(monkeypatch):
    monkeypatch.delenv("PARC_M3_HARDWARE_PROFILE", raising=False)
    _fake_nvidia_smi(
        monkeypatch,
        name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        memory_mib=97887,
    )
    with pytest.raises(RuntimeError, match="requires NVIDIA A100"):
        m3_batch_probe_common.gpu_info()


def test_explicit_organizer_profile_allows_raw_blackwell_identity(monkeypatch):
    monkeypatch.setenv("PARC_M3_HARDWARE_PROFILE", "organizer_rtx_pro_6000_blackwell")
    _fake_nvidia_smi(
        monkeypatch,
        name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        memory_mib=97887,
    )
    name, memory = m3_batch_probe_common.gpu_info()
    assert "RTX PRO 6000" in name
    assert memory == 97887
    profile = m3_hardware_guard.validate_hardware(name, memory)
    assert profile.name == "organizer_rtx_pro_6000_blackwell"


def test_organizer_profile_rejects_small_slice(monkeypatch):
    monkeypatch.setenv("PARC_M3_HARDWARE_PROFILE", "organizer_rtx_pro_6000_blackwell")
    _fake_nvidia_smi(
        monkeypatch,
        name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        memory_mib=49152,
    )
    name, memory = m3_batch_probe_common.gpu_info()
    with pytest.raises(RuntimeError, match="hardware profile mismatch"):
        m3_hardware_guard.validate_hardware(name, memory)
