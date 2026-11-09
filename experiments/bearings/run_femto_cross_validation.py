"""Held-out FEMTO bearings: Full, No age, Condition only, Indicator Rule and Reactive in five outer folds.

    python experiments/bearings/run_femto_cross_validation.py --data-root DATA --output outputs/bearings-femto --jobs 8
"""
from common import main

if __name__ == "__main__":
    raise SystemExit(main("femto_cross_validation"))
