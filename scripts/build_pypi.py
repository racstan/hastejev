#!/usr/bin/env python3
"""Prepare PyPI metadata (strip HF YAML frontmatter from README) and build dist/."""
from __future__ import annotations

import pathlib
import subprocess
import sys


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip("\n")
    return text


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    stripped = strip_frontmatter(readme)
    (root / "README.pypi.md").write_text(stripped, encoding="utf-8")

    pyproject = root / "pyproject.toml"
    t = pyproject.read_text(encoding="utf-8")
    t = t.replace('readme = "README.md"', 'readme = "README.pypi.md"')
    if "[tool.setuptools.packages.find]" not in t:
        t += """
[tool.setuptools.packages.find]
include = ["hastejev*"]
exclude = ["tests*", "benchmarks*", "scripts*", "docs*", "kaggle_kernel*"]

[tool.setuptools]
zip-safe = false
"""
    pyproject.write_text(t, encoding="utf-8")

    setup = root / "setup.py"
    s = setup.read_text(encoding="utf-8")
    old = 'long_description=open("README.md", encoding="utf-8").read() if open("README.md", encoding="utf-8") else "",'
    new = 'long_description=open("README.pypi.md", encoding="utf-8").read(),'
    if old in s:
        setup.write_text(s.replace(old, new), encoding="utf-8")

    for d in ("dist", "build"):
        p = root / d
        if p.exists():
            for child in p.iterdir():
                child.unlink()
            p.rmdir()
    for egg in root.glob("*.egg-info"):
        for child in egg.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(egg.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        egg.rmdir()

    print("README.pypi.md bytes", len(stripped.encode("utf-8")))
    cmd = [sys.executable, "-m", "build"]
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
