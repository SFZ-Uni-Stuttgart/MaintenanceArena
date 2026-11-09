"""MaintenanceArena: Gymnasium environments for maintenance timing under an intervention delay."""
from gymnasium.envs.registration import register, registry

__version__ = "1.0.0"

ENVIRONMENTS = {
    "maintenanceArena/CoreCycle-v0": "maintenance_arena.core:CoreCycleEnv",
    "maintenanceArena/Gearbox-v0": "maintenance_arena.gearbox:GearboxEnv",
    "maintenanceArena/Resource-v0": "maintenance_arena.resource:ResourceEnv",
    "maintenanceArena/ResourceLifetime-v0": "maintenance_arena.resource_lifetime:ResourceLifetimeEnv",
}

for _identifier, _entry in ENVIRONMENTS.items():
    if _identifier not in registry:
        register(id=_identifier, entry_point=_entry)


def make(profile="paper-core-continuous-v1", *, config=None, render_mode=None):
    """Create the Gymnasium environment of a versioned profile, with optional overrides."""
    import gymnasium as gym
    from .profiles import resolve_profile
    resolved = resolve_profile(profile, config)
    env = gym.make(resolved["environment_id"], config=resolved["env_config"], render_mode=render_mode)
    env.unwrapped.arena_profile = resolved
    return env

