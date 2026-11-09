"""Identity freeze of the v1 tasks: hashes, contracts, bytes and short golden runs.

Every value below was computed when the v1 profiles were defined. Documentation,
packaging and tooling work must leave all of them unchanged. A failure here means
that a change altered task identity, a checkpoint binding or simulation behavior.
Such a change needs a new profile ID; do not update these constants to make the
test pass.
"""
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import pytest

from maintenance_arena import gearbox_filter, gearbox_model
from maintenance_arena.core import validate_config
from maintenance_arena.evaluation import evaluate
from maintenance_arena.gearbox import GearboxEnv
from maintenance_arena.profiles import all_profiles, resolve_profile
from maintenance_arena.records import canonical_hash


PROJECT = Path(__file__).resolve().parents[1]
CORE_PROFILES = ("paper-core-continuous-v1", "paper-core-discrete-v1")
CUSTOM_PPO_PROFILES = (*CORE_PROFILES, "paper-gearbox-v1")

# resolve_profile(name)["configuration_sha256"]: the identity of each v1 profile.
PROFILE_SHA256 = {
    "paper-core-continuous-v1": "9b273f610bed7b3b146b801b1abbbd4fe57e67dcc62842e86a0b468e5a9c5980",
    "paper-core-discrete-v1": "affbc44412781a564b10fc7dd2aca01b64ba77649df04d85c0197dcb556532b0",
    "paper-gearbox-v1": "2cfa1482400fc1d9608c538a515cd53b16136a7298bf11abba0d4f78062766fa",
    "paper-resource-v1": "667aa7d03e0d1caaebbcacb11f3c64265dca6b87a75664ee399530f1bc772aae",
    "paper-resource-lifetime-v1": "64db221cc93368abe154eb88a3a335cde3f5dc4d3162114fef2d32bb4027afc2",
}

# Default configurations. A new key in any of them changes the hash.
CORE_DEFAULT_CONFIG_SHA256 = "c209bb132dc033942b5b196573298acfd41466287c19621a94fed82a82006e6e"
GEARBOX_DEFAULT_CONFIG_SHA256 = "f223feb513f7aa0e8e8d59e664b81f3b70902f1daa844f85ee34749edfe9dde3"
RESOURCE_DEFAULT_CONFIG_SHA256 = "46830df2803fe757fe3a31873171951cf1b1227a66eb8a32f35b3bab27964016"

# The validated environment configuration that custom PPO checkpoints bind
# as env_config_sha256 (learning/checkpoints.py).
CHECKPOINT_ENV_CONFIG_SHA256 = {
    "paper-core-continuous-v1": "c209bb132dc033942b5b196573298acfd41466287c19621a94fed82a82006e6e",
    "paper-core-discrete-v1": "f77d106a75c90bcfd98a76dacba790bbc0d9de94f31e5e4820e78114ce8d17ec",
    "paper-gearbox-v1": "f223feb513f7aa0e8e8d59e664b81f3b70902f1daa844f85ee34749edfe9dde3",
}

# canonical_hash(recipe_config(profile)); checkpoints bind it as recipe_sha256.
RECIPE_SHA256 = {
    "paper-core-continuous-v1": "20e6615a65e30b20c481011c1dcd2789e5fb537b95cc756eda58f5c68f8e95e1",
    "paper-core-discrete-v1": "20e6615a65e30b20c481011c1dcd2789e5fb537b95cc756eda58f5c68f8e95e1",
    "paper-gearbox-v1": "a14860793c0acbbd2d2371c6512d65a23a2a938701c48cfd0a515ee17983bb12",
}

# Observation contracts. Core and gearbox values equal the checkpoint field
# observation_sha256. The resource bounds include infinity, so their contract
# writes each bound as repr(float) before hashing.
OBSERVATION_SHA256 = {
    "paper-core-continuous-v1": "249c27d5fb39dc9978cf49037dc00f7a25bdbd217f42add3f771ad3ec097ef34",
    "paper-core-discrete-v1": "249c27d5fb39dc9978cf49037dc00f7a25bdbd217f42add3f771ad3ec097ef34",
    "paper-gearbox-v1": "b6335382efddd3f6138a91b97d790e1792fd8c6f721da3ca6efdcdae17b2c2be",
    "paper-resource-v1": "72166eb28cbf9e6d25d477a497728daba352b1b04ee377271523e42f277436f2",
    "paper-resource-lifetime-v1": "b3583aa6997cdf50e1ad17cda0a720eb1d44f7ceeebe6d41de49944abf502d5f",
}

# Byte SHA-256 of the files that keep CRLF line endings. Their hashes are part of
# the recorded implementation identity, so these bytes must survive every checkout.
CRLF_FILE_SHA256 = {
    "src/maintenance_arena/data/profiles.json": "b7d75c30cded8ba111be7db60ba43f26e86d310bd76a417acd30078e4e87a2c3",
    "src/maintenance_arena/gearbox_filter.py": "fd5f23370cf0ccbc2f1f146702ad283e5031c517fcc1ea04429a0edfbf23674d",
    "src/maintenance_arena/gearbox_model.py": "505bcbd8e90acd6ec6ff8f92ec246f00328f17a4abc0103a5106fc76fdf4a98a",
}

# Short golden runs: evaluate(profile, policies=<built-in rules>, seed=GOLDEN_SEED,
# episodes=4, calendar_steps=240). Rows: policy, calendar steps, failures,
# planned interventions, corrective interventions, raw profit,
# mean profit per 1,000 calendar steps, mean normalized discarded life.
GOLDEN_SEED = 20261002
GOLDEN_SCENARIO_SEEDS = [1350759447, 2618224332, 898922451, 2295867616]
GOLDEN = {
    "paper-core-continuous-v1": (
        ("reactive", 553, 4, 0, 4, 103095.0, 186298.81222498656, 0.0),
        ("age", 408, 0, 4, 0, 94120.0, 230686.27450980392, 0.2801779149159664),
        ("health", 391, 0, 4, 0, 89785.0, 228708.91529428118, 0.31110333508403365),
        ("projection", 547, 0, 4, 0, 129565.0, 236836.0788914224, 0.018731223739495796),
        ("immediate", 76, 0, 4, 0, 9460.0, 124473.68421052632, 0.9090751050420168),
    ),
    "paper-core-discrete-v1": (
        ("reactive", 520, 4, 0, 4, 94680.0, 181343.94131039717, 0.0),
        ("age", 408, 0, 4, 0, 94120.0, 230686.27450980392, 0.2254154265873016),
        ("health", 365, 0, 4, 0, 83155.0, 226696.38661733526, 0.3177703373015873),
        ("projection", 518, 0, 4, 0, 122170.0, 235658.67791557396, 0.012121775793650792),
        ("immediate", 76, 0, 4, 0, 9460.0, 124473.68421052632, 0.902157738095238),
    ),
    "paper-gearbox-v1": (
        ("reactive_normal", 960, 4, 0, 4, 200874.75, 209244.53125, None),
        ("reactive_derate", 960, 4, 0, 4, 194997.0, 203121.875, None),
        ("normal_q96", 960, 0, 4, 0, 228594.25, 238119.01041666666, None),
        ("derate_q89", 960, 0, 4, 0, 222997.0, 232288.54166666666, None),
        ("separable_q96", 960, 0, 4, 0, 228581.5, 238105.72916666666, None),
        ("dp", 960, 0, 4, 0, 228568.75, 238092.44791666666, None),
    ),
    "paper-resource-v1": (
        ("reactive", 960, 12, 0, 12, 211681.79999999978, 220501.87499999977, None),
        ("health", 960, 0, 17, 0, 248889.69999999998, 259260.10416666666, None),
        ("rolling", 960, 0, 16, 0, 250926.1999999999, 261381.45833333323, None),
    ),
    "paper-resource-lifetime-v1": (
        ("reactive", 960, 12, 0, 12, 214863.05000000042, 223815.67708333378, None),
        ("age", 960, 1, 18, 1, 246990.80000000008, 257282.0833333334, None),
        ("health", 960, 0, 18, 0, 239888.35000000006, 249883.69791666672, None),
        ("rolling", 960, 0, 16, 0, 250926.64999999988, 261381.9270833332, None),
    ),
}
# Floating sums may differ in the last bits across platforms and libm builds.
# Any behavioral change moves the counts or the profit far beyond this.
FLOAT_TOLERANCE = {"rel": 1e-9, "abs": 1e-6}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _line_endings(data: bytes) -> str:
    crlf = data.count(b"\r\n")
    bare_lf = data.count(b"\n") - crlf
    return f"{crlf} CRLF and {bare_lf} bare LF line endings"


def _observation_contract(env):
    """Same mapping as learning.checkpoints.observation_contract, without torch."""
    return {"shape": list(env.observation_space.shape), "dtype": str(env.observation_space.dtype),
            "names": list(env.observation_names), "actions": int(env.action_space.n),
            "lower": env.observation_space.low.tolist(), "upper": env.observation_space.high.tolist()}


def _resource_observation_contract(env):
    contract = _observation_contract(env)
    contract["lower"] = [repr(float(value)) for value in env.observation_space.low]
    contract["upper"] = [repr(float(value)) for value in env.observation_space.high]
    return contract


def _make(name):
    profile = resolve_profile(name)
    return gym.make(profile["environment_id"], config=profile["env_config"])


def test_frozen_profiles_are_still_published():
    assert set(PROFILE_SHA256) <= set(all_profiles())


@pytest.mark.parametrize("name", sorted(PROFILE_SHA256))
def test_profile_configuration_hash_is_frozen(name):
    profile = resolve_profile(name)
    assert profile["id"] == name and not profile["modified"]
    assert profile["configuration_sha256"] == PROFILE_SHA256[name]


def test_default_configuration_hashes_are_frozen():
    from dataclasses import asdict
    from maintenance_arena.resource import ResourceMaintenanceConfig

    assert canonical_hash(validate_config(None)) == CORE_DEFAULT_CONFIG_SHA256
    env = GearboxEnv()
    try:
        assert canonical_hash(env.config) == GEARBOX_DEFAULT_CONFIG_SHA256
    finally:
        env.close()
    assert canonical_hash(asdict(ResourceMaintenanceConfig())) == RESOURCE_DEFAULT_CONFIG_SHA256


@pytest.mark.parametrize("name", CUSTOM_PPO_PROFILES)
def test_checkpoint_bound_environment_configuration_is_frozen(name):
    configuration = resolve_profile(name)["env_config"]
    if name in CORE_PROFILES:
        validated = validate_config(configuration)
    else:
        env = GearboxEnv(configuration)
        validated = env.config
        env.close()
    assert canonical_hash(validated) == CHECKPOINT_ENV_CONFIG_SHA256[name]


@pytest.mark.parametrize("name", sorted(OBSERVATION_SHA256))
def test_observation_contract_is_frozen(name):
    env = _make(name)
    try:
        contract = (_resource_observation_contract(env.unwrapped) if "Resource" in resolve_profile(name)["environment_id"]
                    else _observation_contract(env.unwrapped))
    finally:
        env.close()
    assert canonical_hash(contract) == OBSERVATION_SHA256[name]


@pytest.mark.parametrize("name", CUSTOM_PPO_PROFILES)
def test_checkpoint_observation_and_recipe_bindings_are_frozen(name):
    pytest.importorskip("torch")
    from maintenance_arena.learning.checkpoints import observation_contract
    from maintenance_arena.learning.ppo import recipe_config

    env = _make(name)
    try:
        assert observation_contract(env.unwrapped) == _observation_contract(env.unwrapped)
    finally:
        env.close()
    assert canonical_hash(recipe_config(name)) == RECIPE_SHA256[name]


@pytest.mark.parametrize("relative", sorted(CRLF_FILE_SHA256))
def test_crlf_file_bytes_are_frozen(relative):
    source = PROJECT / relative
    if not source.is_file():
        pytest.skip("source checkout is not available next to the tests")
    data = source.read_bytes()
    assert _sha256(data) == CRLF_FILE_SHA256[relative], f"{relative}: {_line_endings(data)}"


def test_imported_crlf_package_files_are_the_frozen_bytes():
    from importlib.resources import files

    imported = {
        "src/maintenance_arena/data/profiles.json": files("maintenance_arena").joinpath("data/profiles.json").read_bytes(),
        "src/maintenance_arena/gearbox_filter.py": Path(gearbox_filter.__file__).read_bytes(),
        "src/maintenance_arena/gearbox_model.py": Path(gearbox_model.__file__).read_bytes(),
    }
    for relative, data in imported.items():
        assert _sha256(data) == CRLF_FILE_SHA256[relative], f"{relative}: {_line_endings(data)}"


def test_provenance_is_the_refresh_serialization_of_its_inventory():
    source = PROJECT / "provenance.json"
    if not source.is_file():
        pytest.skip("source checkout is not available next to the tests")
    data = source.read_bytes()
    recorded = json.loads(data)
    # The file is exactly the refresh-current serialization with LF line endings.
    expected = (json.dumps(recorded, indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert data == expected, f"provenance.json: {_line_endings(data)}"
    assert set(recorded) == {"schema", "meaning", "current_implementation"}
    assert recorded["schema"] == "maintenance_arena_provenance_v2"
    files = recorded["current_implementation"]["files"]
    for relative, digest in CRLF_FILE_SHA256.items():
        assert files[relative] == digest, relative


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_evaluation_summary(name):
    rows = GOLDEN[name]
    policies = [row[0] for row in rows]
    result = evaluate(resolve_profile(name), policies=policies, seed=GOLDEN_SEED,
                      episodes=len(GOLDEN_SCENARIO_SEEDS), calendar_steps=240)
    assert result["fixture"]["seeds"] == GOLDEN_SCENARIO_SEEDS
    assert [summary["policy"] for summary in result["summary"]] == policies
    for summary, row in zip(result["summary"], rows, strict=True):
        _, calendar, failures, planned, corrective, raw_profit, mean_rate, discarded = row
        assert summary["independent_synthetic_scenarios"] == len(GOLDEN_SCENARIO_SEEDS)
        assert summary["corrected_proposals"] == 0
        assert (summary["actual_calendar_steps"], summary["failures"],
                summary["planned_interventions"], summary["corrective_interventions"]) == (
                    calendar, failures, planned, corrective), summary["policy"]
        assert summary["raw_profit"] == pytest.approx(raw_profit, **FLOAT_TOLERANCE), summary["policy"]
        assert summary["mean_profit_per_1000_calendar_steps"] == pytest.approx(mean_rate, **FLOAT_TOLERANCE), summary["policy"]
        if discarded is None:
            assert summary["mean_normalized_discarded_life"] is None
        else:
            assert summary["mean_normalized_discarded_life"] == pytest.approx(discarded, **FLOAT_TOLERANCE), summary["policy"]
