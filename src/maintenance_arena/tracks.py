"""Named benchmark tracks: fixed test and calibration scenarios on the unchanged v1 profiles.

A track fixes everything that a benchmark result depends on except the agent: the
profile (by its ID, unchanged), the calendar budget of each scenario, the references,
the core rules that are calibrated, what an agent observes, and two explicit seed
lists. The test seeds are the scenarios that are scored. The calibration seeds are
the scenarios on which the core rules are calibrated; agents may be tuned and
selected on them as well, but never on the test seeds.

The packaged tracks are in ``data/benchmark_tracks.json``. Both seed lists of a
track are derived from its documented ``root_seed`` with NumPy's ``SeedSequence``:

* test scenario ``i`` (``i < test_scenarios``) has the seed
  ``SeedSequence([root_seed, i]).generate_state(1, dtype=uint32)[0]``. These are the
  scenarios that ``evaluate`` and ``benchmark`` derive from ``--seed root_seed
  --episodes test_scenarios``.
* calibration scenario ``i`` has the seed
  ``SeedSequence(root_seed, spawn_key=(1, i)).generate_state(1, dtype=uint32)[0]``,
  as in ``calibration.calibration_seeds``; a test seed or a repeated seed is skipped,
  so the two lists are disjoint.

The lists are stored explicitly, so a track keeps its scenarios even if a later NumPy
release derived different numbers; ``track_seeds`` re-derives them for a check.
``benchmark.run_track`` scores agents on a track and ``make_track`` builds the
environment of a track for training, with the track's calendar budget and agent
observation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
import json
from pathlib import Path

import numpy as np

from . import evaluation
from .profiles import all_profiles, resolve_profile
from .records import canonical_hash


SCHEMA = "maintenance_arena_benchmark_tracks_v1"
DEFAULT_TRACK = "core-continuous-v1"
# What an agent receives on a track, as written in the information column.
AGENT_OBSERVATIONS = {
    "full": "observation and info",
    "zero_deadline": "observation with the deadline coordinate (index 3) set to zero, and info",
}
TEST_DERIVATION = ("SeedSequence([root_seed, i]).generate_state(1, dtype=uint32)[0] for i = 0 .. test_scenarios - 1; "
                   "the scenarios that evaluate and benchmark derive from --seed root_seed --episodes test_scenarios")
CALIBRATION_DERIVATION = ("SeedSequence(root_seed, spawn_key=(1, i)).generate_state(1, dtype=uint32)[0] for "
                          "i = 0, 1, ... until calibration_scenarios seeds are chosen; a test seed or a seed already "
                          "chosen is skipped (maintenance_arena.calibration.calibration_seeds)")
FIELDS = ("name", "description", "profile", "profile_configuration_sha256", "environment_id", "task",
          "agent_observation", "calendar_steps", "root_seed", "test_scenarios", "calibration_scenarios",
          "references", "calibrated_rules", "test_seeds", "calibration_seeds")
_SEED_LISTS = ("test_seeds", "calibration_seeds")


def track_seeds(root_seed, test_scenarios, calibration_scenarios) -> tuple[list[int], list[int]]:
    """Derive the test and calibration seeds of a track from its root seed (see the module docstring)."""
    from .calibration import calibration_seeds
    root = evaluation._integer(root_seed, "root_seed")
    count = evaluation._integer(test_scenarios, "test_scenarios", 1)
    test = [int(np.random.SeedSequence([root, index]).generate_state(1, dtype=np.uint32)[0]) for index in range(count)]
    if len(set(test)) != len(test):
        raise ValueError(f"Root seed {root} gives a repeated test seed; choose another root seed")
    return test, calibration_seeds(root, calibration_scenarios, exclude=test)


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"Track {name} must be a nonempty string without surrounding whitespace")
    return value


def _names(values, name):
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f"Track {name} must be a list of names")
    result = tuple(_text(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"Track {name} contains a name twice")
    return result


def _seeds(values, name, expected):
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f"Track {name} must be a list of nonnegative integer seeds")
    result = tuple(evaluation._integer(value, f"track {name} entry") for value in values)
    if len(result) != expected:
        raise ValueError(f"Track {name} holds {len(result)} seeds, but the track declares {expected}")
    if len(set(result)) != len(result):
        raise ValueError(f"Track {name} contains a seed twice")
    return result


@dataclass(frozen=True)
class Track:
    """One benchmark track; ``to_dict`` and ``from_dict`` give its JSON form."""

    name: str
    description: str
    profile: str
    profile_configuration_sha256: str
    environment_id: str
    task: str
    agent_observation: str
    calendar_steps: int
    root_seed: int
    test_scenarios: int
    calibration_scenarios: int
    references: tuple[str, ...]
    calibrated_rules: tuple[str, ...]
    test_seeds: tuple[int, ...] = field(repr=False)
    calibration_seeds: tuple[int, ...] = field(repr=False)

    def to_dict(self) -> dict:
        """The JSON form of the track, with the fields in the order of ``FIELDS``."""
        return {name: list(getattr(self, name)) if isinstance(getattr(self, name), tuple) else getattr(self, name)
                for name in FIELDS}

    @property
    def sha256(self) -> str:
        """Hash of the complete track definition, including both seed lists."""
        return canonical_hash(self.to_dict())

    def summary(self) -> dict:
        """The track definition without the seed lists, plus its ``sha256``."""
        result = {name: value for name, value in self.to_dict().items() if name not in _SEED_LISTS}
        result["sha256"] = self.sha256
        return result

    @property
    def agent_information(self) -> str:
        """What an agent receives on this track, as written in the benchmark table."""
        return AGENT_OBSERVATIONS[self.agent_observation]

    def resolved_profile(self) -> dict:
        """The track's profile, checked against the configuration hash that the track records."""
        profile = resolve_profile(self.profile)
        if profile["configuration_sha256"] != self.profile_configuration_sha256:
            raise ValueError(f"Track {self.name} was defined for another configuration of {self.profile} "
                             f"(sha256 {self.profile_configuration_sha256[:12]}, now "
                             f"{profile['configuration_sha256'][:12]}); its results would not be comparable")
        return profile

    def fixture(self) -> dict:
        """The test scenarios as an evaluation fixture (``evaluate --fixture`` replays it)."""
        return evaluation.make_fixture(self.resolved_profile(), list(self.test_seeds), calendar_steps=self.calendar_steps)

    @classmethod
    def from_dict(cls, data) -> "Track":
        """Validate a track's JSON form; the profile must be a packaged, unchanged profile ID."""
        from .benchmark import PRIMARY_REFERENCES, benchmark_entries, task_of
        from .calibration import RULES
        if not isinstance(data, Mapping):
            raise ValueError("A track must be a JSON object")
        missing, unknown = [name for name in FIELDS if name not in data], sorted(set(data) - set(FIELDS))
        if missing or unknown:
            raise ValueError(f"Track fields differ from the track format: missing {missing}, unknown {unknown}")
        name = _text(data["name"], "name")
        if name.lower().endswith(".json") or any(character in name for character in "/\\:"):
            raise ValueError(f"Track name {name!r} must not look like a file path")
        if not isinstance(data["description"], str):
            raise ValueError("Track description must be a string")
        profile_id = _text(data["profile"], "profile")
        if profile_id not in all_profiles():
            raise ValueError(f"Track {name} names the unknown profile {profile_id!r}; tracks use packaged profile IDs "
                             "without overrides")
        values = dict(data)
        for key, minimum in (("calendar_steps", 1), ("root_seed", 0), ("test_scenarios", 1),
                             ("calibration_scenarios", 1)):
            values[key] = evaluation._integer(data[key], f"track {key}", minimum)
        track = cls(name=name, description=data["description"], profile=profile_id,
                    profile_configuration_sha256=_text(data["profile_configuration_sha256"],
                                                       "profile_configuration_sha256"),
                    environment_id=_text(data["environment_id"], "environment_id"), task=_text(data["task"], "task"),
                    agent_observation=_text(data["agent_observation"], "agent_observation"),
                    calendar_steps=values["calendar_steps"], root_seed=values["root_seed"],
                    test_scenarios=values["test_scenarios"], calibration_scenarios=values["calibration_scenarios"],
                    references=_names(data["references"], "references"),
                    calibrated_rules=_names(data["calibrated_rules"], "calibrated_rules"),
                    test_seeds=_seeds(data["test_seeds"], "test_seeds", values["test_scenarios"]),
                    calibration_seeds=_seeds(data["calibration_seeds"], "calibration_seeds",
                                             values["calibration_scenarios"]))
        profile = track.resolved_profile()
        if track.environment_id != profile["environment_id"]:
            raise ValueError(f"Track {name}: {profile_id} uses {profile['environment_id']}, not {track.environment_id}")
        task = task_of(profile)
        if track.task != task:
            raise ValueError(f"Track {name}: the task of {profile_id} is {task!r}, not {track.task!r}")
        if track.agent_observation not in AGENT_OBSERVATIONS:
            raise ValueError(f"Track {name}: agent_observation must be one of {', '.join(AGENT_OBSERVATIONS)}")
        if track.agent_observation == "zero_deadline" and task != "core":
            raise ValueError(f"Track {name}: zero_deadline applies to the core task only")
        primary = PRIMARY_REFERENCES[task]
        if primary not in track.references:
            raise ValueError(f"Track {name} must list its primary reference {primary!r} among the references")
        if "ppo" in track.references:
            raise ValueError(f"Track {name}: ppo needs a trained model and cannot be a track reference; "
                             "score a model as an agent instead")
        benchmark_entries(profile, references=list(track.references))
        for rule in track.calibrated_rules:
            if task != "core" or rule not in RULES or rule not in track.references:
                raise ValueError(f"Track {name}: calibrated_rules must name core rules among the references "
                                 f"({', '.join(RULES)}); got {rule!r}")
        overlap = set(track.test_seeds) & set(track.calibration_seeds)
        if overlap:
            raise ValueError(f"Track {name}: {len(overlap)} seeds are both test and calibration seeds, "
                             f"e.g. {min(overlap)}; the two lists must be disjoint")
        return track


def define_track(name, profile, *, root_seed, test_scenarios, calibration_scenarios, calendar_steps=720,
                 references=None, calibrated_rules=None, agent_observation="full", description="") -> Track:
    """Define a track on a packaged profile, deriving both seed lists from ``root_seed``.

    references: default ``benchmark.default_references(profile)``, the primary reference
        and every built-in policy except ``ppo``.
    calibrated_rules: default, for a core profile, the rules ``age``, ``health`` and
        ``projection`` among the references; none for the other tasks.
    """
    from .benchmark import default_references, task_of
    from .calibration import RULES
    if not isinstance(profile, str):
        raise ValueError("A track names a packaged profile ID; overrides would change the task")
    resolved = resolve_profile(profile)
    task = task_of(resolved)
    references = list(default_references(resolved) if references is None else references)
    if calibrated_rules is None:
        calibrated_rules = [rule for rule in RULES if rule in references] if task == "core" else []
    test, calibration = track_seeds(root_seed, test_scenarios, calibration_scenarios)
    return Track.from_dict({
        "name": name, "description": description, "profile": profile,
        "profile_configuration_sha256": resolved["configuration_sha256"],
        "environment_id": resolved["environment_id"], "task": task, "agent_observation": agent_observation,
        "calendar_steps": calendar_steps, "root_seed": root_seed, "test_scenarios": test_scenarios,
        "calibration_scenarios": calibration_scenarios, "references": references,
        "calibrated_rules": list(calibrated_rules), "test_seeds": test, "calibration_seeds": calibration})


def tracks_document(tracks, default_track=None) -> dict:
    """The JSON document of a tracks file; ``default_track`` defaults to the first track."""
    tracks = [track if isinstance(track, Track) else Track.from_dict(track) for track in tracks]
    if not tracks:
        raise ValueError("A tracks file needs at least one track")
    names = [track.name for track in tracks]
    if len(set(names)) != len(names):
        raise ValueError("Track names must be unique within a tracks file")
    default = names[0] if default_track is None else default_track
    if default not in names:
        raise ValueError(f"default_track {default!r} is not among the tracks {names}")
    return {"schema": SCHEMA, "default_track": default,
            "seed_derivation": {"test": TEST_DERIVATION, "calibration": CALIBRATION_DERIVATION},
            "tracks": [track.to_dict() for track in tracks]}


def dumps_tracks(document) -> str:
    """Deterministic JSON text of a tracks document: two-space indentation, ten seeds per line."""
    def scalar(value):
        return json.dumps(value, ensure_ascii=False)

    def track_lines(track):
        lines = []
        for index, name in enumerate(FIELDS):
            value = track[name]
            if name in _SEED_LISTS:
                rows = [", ".join(str(seed) for seed in value[start:start + 10]) for start in range(0, len(value), 10)]
                text = "[\n" + ",\n".join(f"        {row}" for row in rows) + "\n      ]" if rows else "[]"
            else:
                text = scalar(value)
            lines.append(f'      {scalar(name)}: {text}' + ("," if index < len(FIELDS) - 1 else ""))
        return "    {\n" + "\n".join(lines) + "\n    }"

    derivation = document["seed_derivation"]
    return ("{\n"
            f'  "schema": {scalar(document["schema"])},\n'
            f'  "default_track": {scalar(document["default_track"])},\n'
            '  "seed_derivation": {\n'
            f'    "test": {scalar(derivation["test"])},\n'
            f'    "calibration": {scalar(derivation["calibration"])}\n'
            "  },\n"
            '  "tracks": [\n' + ",\n".join(track_lines(track) for track in document["tracks"]) + "\n  ]\n}\n")


def write_tracks(path, tracks, *, default_track=None) -> Path:
    """Write a tracks file for custom tracks; an existing file is never overwritten."""
    target = Path(path)
    text = dumps_tracks(tracks_document(tracks, default_track))
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return target


def _read_document(path=None):
    if path is None:
        text = files("maintenance_arena").joinpath("data/benchmark_tracks.json").read_text(encoding="utf-8")
        origin = "the packaged benchmark_tracks.json"
    else:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise ValueError(f"Tracks file does not exist: {path}") from error
        origin = str(path)
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{origin} is not valid JSON: {error}") from error
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        raise ValueError(f"{origin} needs schema {SCHEMA}")
    if set(document) != {"schema", "default_track", "seed_derivation", "tracks"}:
        raise ValueError(f"{origin} must hold exactly schema, default_track, seed_derivation and tracks")
    entries = document["tracks"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{origin} must list at least one track")
    tracks = {}
    for entry in entries:
        track = Track.from_dict(entry)
        if track.name in tracks:
            raise ValueError(f"{origin} defines track {track.name} twice")
        tracks[track.name] = track
    if document["default_track"] not in tracks:
        raise ValueError(f"{origin}: default_track {document['default_track']!r} is not among its tracks")
    return document, tracks


def load_tracks(path=None) -> dict[str, Track]:
    """The tracks of a tracks file by name; without ``path`` the packaged tracks."""
    return _read_document(path)[1]


def track_names() -> tuple[str, ...]:
    """Names of the packaged tracks, in file order."""
    return tuple(load_tracks())


def load_track(track=None) -> Track:
    """Resolve a track: None (the default track), a packaged name, a tracks file, a ``Track`` or its JSON form.

    A tracks file (``*.json``, the format of ``data/benchmark_tracks.json``) gives its
    only track, or its ``default_track`` when it holds several.
    """
    if track is None:
        return load_tracks()[DEFAULT_TRACK]
    if isinstance(track, Track):
        return track
    if isinstance(track, Mapping):
        return Track.from_dict(track)
    if isinstance(track, Path) or (isinstance(track, str) and track.lower().endswith(".json")):
        document, tracks = _read_document(track)
        return tracks[document["default_track"]]
    if isinstance(track, str):
        packaged = load_tracks()
        if track in packaged:
            return packaged[track]
        raise ValueError(f"Unknown track {track!r}; choose from {', '.join(packaged)}, or give a tracks JSON file")
    raise ValueError("track must be a track name, a tracks JSON file, a Track or a track mapping")


def make_track(track=None, *, render_mode=None):
    """The Gymnasium environment of a track, for training an agent outside the test scenarios.

    It is ``gymnasium.make`` of the track's environment with the profile's configuration
    (and, for the resource tasks, the track's horizon), wrapped in ``CalendarBudget`` with
    the track's calendar budget and, on a zero-deadline track, in ``ZeroDeadline``. The
    resolved profile is attached as ``env.unwrapped.arena_profile``. Seed it with your own
    seeds; the track's test seeds are for scoring only.
    """
    import gymnasium as gym
    from .wrappers import CalendarBudget, ZeroDeadline
    spec = load_track(track)
    profile = spec.resolved_profile()
    env = gym.make(spec.environment_id, config=evaluation._environment_configuration(profile, spec.calendar_steps),
                   render_mode=render_mode)
    env.unwrapped.arena_profile = profile
    env = CalendarBudget(env, spec.calendar_steps)
    if spec.agent_observation == "zero_deadline":
        env = ZeroDeadline(env)
    return env


__all__ = ["AGENT_OBSERVATIONS", "DEFAULT_TRACK", "FIELDS", "SCHEMA", "Track", "define_track", "dumps_tracks",
           "load_track", "load_tracks", "make_track", "track_names", "track_seeds", "tracks_document",
           "write_tracks"]
