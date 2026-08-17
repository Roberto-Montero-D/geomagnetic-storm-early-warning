# Geomagnetic Storm Early Warning System

This project builds a causally correct, operationally useful, and scientifically interpretable early warning system for geomagnetic storms (Kp >= 5).

## Key Features

- **Event-in-Window Target:** Predicts whether a storm will occur within the next H hours
- **Causal Features:** All features audited for data leakage
- **Walk-Forward Validation:** Robust temporal validation
- **Operational Metrics:** Event Recall, FAR/day, Lead Time

## Documentation

- [Master Protocol](MASTER_PROTOCOL.md) - Complete project protocol (frozen)
- [Data Contract](docs/data_contract.md) - Feature availability audit

## Repository Structure
geomagnetic-storm-early-warning/
├── config/ # Centralized configuration
├── src/ # Source code
│ ├── data/ # Dataset and features
│ ├── definitions/ # Event and alert definitions
│ ├── models/ # Baselines, screening, model selection
│ ├── evaluation/ # OOF threshold, final test
│ └── analysis/ # Error analysis, SHAP, ablation
├── notebooks/ # Jupyter notebooks
├── tests/ # Unit tests
├── results/ # Figures and tables
└── docs/ # Detailed documentation

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run screening: `python scripts/run_screening.py`
3. Run walk-forward: `python scripts/run_walk_forward.py`
4. Run final test: `python scripts/run_final_test.py`

## License

MIT