"""Dependency pins that keep the runner installable from wheels on every supported machine."""
import tomllib
from pathlib import Path


def test_intel_macs_get_a_cryptography_with_wheels():
    deps = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())["project"]["dependencies"]
    assert "cryptography<49; sys_platform == 'darwin' and platform_machine == 'x86_64'" in deps
