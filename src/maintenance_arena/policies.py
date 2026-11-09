"""Policy dispatch uses declared observations and public feasibility only."""
from __future__ import annotations
import importlib
import json
from pathlib import Path, PurePosixPath

from .records import file_hash


def environment_module(profile):
    name = profile["environment_id"].split("/")[1].split("-")[0]
    return importlib.import_module("maintenance_arena." + {
        "CoreCycle": "core", "Gearbox": "gearbox", "Resource": "resource", "ResourceLifetime": "resource_lifetime"
    }[name])


def contained_file(root, entry):
    name = entry["path"]
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Artifact paths must be portable relative paths within the manifest directory")
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path escapes its manifest directory")
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {entry['path']}")
    return path


def load_policy(checkpoint, profile, env):
    path = Path(checkpoint)
    if "Resource" in profile["environment_id"]:
        from .learning.resource_ppo import load_policy as load
        return load(path, profile=profile["base_profile"], env=env.unwrapped)
    expected = None
    if path.is_dir():
        path = path / "final.pt"
    if path.suffix == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("profile") != profile["base_profile"]:
            raise ValueError("Checkpoint profile does not match this task")
        entry = manifest["files"]["checkpoint"]
        expected = entry["sha256"]
        path = contained_file(path.parent, entry)
    from .learning.checkpoints import load_policy as load
    return load(path, profile=profile["base_profile"], config=profile["env_config"], expected_sha256=expected)


class Policy:
    def __init__(self, name, profile, env, checkpoint=None):
        if name not in profile["policies"]:
            raise ValueError(f"Policy {name!r} is not supported by {profile['base_profile']}; "
                             f"choose from {', '.join(profile['policies'])}")
        self.name, self.profile, self.env = name, profile, env
        self.module = environment_module(profile)
        self.learned = None
        if name == "ppo":
            if checkpoint is None:
                raise ValueError("PPO evaluation requires --checkpoint; no model is downloaded or trained implicitly")
            self.learned = load_policy(checkpoint, profile, env)

    def __call__(self, observation, info):
        if self.learned is not None:
            if "Resource" in self.profile["environment_id"]:
                return int(self.learned.predict(observation, action_mask=self.env.unwrapped.action_masks()))
            return int(self.learned.predict(observation))
        return int(self.module.policy_action(self.name, self.env.unwrapped, observation,
                   **self.profile["policy_defaults"].get(self.name, {})))

    def close(self):
        if self.learned is not None and hasattr(self.learned, "close"):
            self.learned.close()
