# Phase 8 Software Environment Freeze

**Purpose:** preserve the numerical/software environment before the single-use
protected Final Test is scored.

The development environment that successfully executed the frozen Phase 6 and
Phase 7 pipeline used:

```text
OS       = Windows
Python   = 3.14.7
NumPy    = 2.5.2
pandas   = 3.0.5
SciPy    = 1.18.0
scikit-learn = 1.9.0
imbalanced-learn = 0.14.2
LightGBM = 4.7.0
XGBoost  = 3.4.1
PyYAML   = 6.0.3
pytest   = 9.1.1
```

`requirements-lock-phase8.txt` pins the direct project dependencies used by the
repository. The original `requirements.txt` remains the general development
dependency specification and is not rewritten.

Before the protected Final Test runner is executed, capture the exact active
environment as an additional audit artifact:

```powershell
python --version
python -m pip freeze --all > results\phase8\environment\pip-freeze.txt
python -m pip list > results\phase8\environment\pip-list.txt
```

Also record:

```powershell
python -c "import sys, platform; print(sys.version); print(platform.platform())" `
  > results\phase8\environment\python-platform.txt
```

The environment record is provenance, not a tuning input. If the environment
cannot reproduce the already-frozen Phase 7 positive-control path, do not open
the Final Test; repair the environment first.
