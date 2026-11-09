"""Replay of run-to-failure bearing recordings (FEMTO/PRONOSTIA and XJTU-SY) for maintenance timing.

The subpackage turns raw vibration files into minute histories, causal condition features
and remaining-time predictions, replays WAIT/ORDER decisions with the core task's
preparation, exchange and prices, and provides the Time rule of the article (forecast
rules), the Indicator rule and the broader rule search, a supervised risk classifier, a bearing PPO recipe
(``maintenance_arena.bearings.ppo``), equal-bearing and pooled metrics, candidate selection
and figure data. No dataset is shipped. Raw files need pandas, the prediction model needs
scikit-learn and PPO needs PyTorch; importing this package imports none of them.
"""
from .analysis import age_clock, median_order, policy_map, policy_responses, transfer_timing
from .classifier import (CLASSIFIER_SETTINGS, ClassifierSelection, calibrate_risk_classifier, classifier_order,
                         fit_classifier, risk_classifier_orders)
from .data import (DATASETS, STATISTICS, BearingHistory, DatasetLayout, acquisition_files,
                   acquisition_statistics, check_layout, dataset_directory, histories, load_statistics, prepare,
                   read_acquisition, to_minutes)
from .economics import ORDER, RECOVERY_CONTRACTS, WAIT, ReplayEconomics, ReplayOutcome, pooled_rate
from .env import BearingReplayEnv
from .features import (INPUT_DESIGNS, OBSERVATION_BOUNDS, OBSERVATION_NAMES, FeatureSettings, InputDesign,
                       decision_series, design_observations, forecaster_features, input_design, policy_observations)
from .forecaster import Forecaster, ForecasterSettings, cross_fit, hashed_folds
from .metrics import (OUTCOME_FIELDS, bearing_metrics, design_summary, outcome_row, paired_bootstrap,
                      pooled_bootstrap, pooled_summary, rank_candidates, read_outcomes, write_outcomes)
from .rules import (RULE_FAMILIES, RuleCandidate, RuleFamily, RuleGrid, apply_rule, calibrate_forecast_rule,
                    calibrate_indicator_rule, family_orders, family_parameters, forecast_rule_order,
                    indicator_candidates, search_rule_families)

__all__ = [
    "age_clock", "median_order", "policy_map", "policy_responses", "transfer_timing",
    "CLASSIFIER_SETTINGS", "ClassifierSelection", "calibrate_risk_classifier", "classifier_order", "fit_classifier",
    "risk_classifier_orders",
    "DATASETS", "STATISTICS", "BearingHistory", "DatasetLayout", "acquisition_files", "acquisition_statistics",
    "check_layout", "dataset_directory", "histories", "load_statistics", "prepare", "read_acquisition", "to_minutes",
    "ORDER", "RECOVERY_CONTRACTS", "WAIT", "ReplayEconomics", "ReplayOutcome", "pooled_rate",
    "BearingReplayEnv",
    "INPUT_DESIGNS", "OBSERVATION_BOUNDS", "OBSERVATION_NAMES", "FeatureSettings", "InputDesign",
    "decision_series", "design_observations", "forecaster_features", "input_design", "policy_observations",
    "Forecaster", "ForecasterSettings", "cross_fit", "hashed_folds",
    "OUTCOME_FIELDS", "bearing_metrics", "design_summary", "outcome_row", "paired_bootstrap", "pooled_bootstrap",
    "pooled_summary", "rank_candidates", "read_outcomes", "write_outcomes",
    "RULE_FAMILIES", "RuleCandidate", "RuleFamily", "RuleGrid", "apply_rule", "calibrate_forecast_rule",
    "calibrate_indicator_rule", "family_orders", "family_parameters", "forecast_rule_order", "indicator_candidates",
    "search_rule_families",
]
