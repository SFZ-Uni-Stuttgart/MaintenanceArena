"""Broader rule search on held-out FEMTO: sixteen families, a cross-validated choice and the held-out choice.

    python experiments/bearings/run_rule_search.py --data-root DATA --output outputs/bearings-rule-search

Writes the rows of every family, of the Cross-validated rule and of the family with the highest
pooled held-out profit; selections.json holds the inner cross-validation scores of every fold.
"""
from common import main

if __name__ == "__main__":
    raise SystemExit(main("rule_search"))
