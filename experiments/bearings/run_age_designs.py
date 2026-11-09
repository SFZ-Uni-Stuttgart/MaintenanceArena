"""Mixed age designs, No policy age and No prediction age (stored as No forecaster age), held-out FEMTO and XJTU-SY.

    python experiments/bearings/run_age_designs.py --data-root DATA --output outputs/bearings-age --jobs 8
"""
from common import main

if __name__ == "__main__":
    raise SystemExit(main("age_designs"))
