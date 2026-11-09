"""Candidate selection: Full, No age and Condition only at 409,600 and 819,200 steps, ranked on held-out FEMTO.

    python experiments/bearings/run_candidate_selection.py --data-root DATA --output outputs/bearings-candidates --jobs 12

Writes selection.json with the ranking by pooled held-out profit next to the outcome rows.
"""
from common import main

if __name__ == "__main__":
    raise SystemExit(main("candidate_selection"))
