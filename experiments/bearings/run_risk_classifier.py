"""Risk classifier on the inputs of Full and No age, on held-out FEMTO and after transfer to XJTU-SY.

    python experiments/bearings/run_risk_classifier.py --data-root DATA --output outputs/bearings-risk-classifier

Writes the rows of both classifiers; selections.json holds the threshold of every fold and of the transfer.
"""
from common import main

if __name__ == "__main__":
    raise SystemExit(main("risk_classifier"))
