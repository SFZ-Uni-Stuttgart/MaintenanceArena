"""Train a Stable-Baselines3 agent on the core task and score it on a benchmark track.

With the rl extra (PyTorch, Stable-Baselines3), run from the repository root:

    python examples/benchmark_your_agent.py

This trains stock PPO through gymnasium.make for 50,000 steps on the CPU, which takes a
few minutes. It scores the model on the default track core-continuous-v1 with its 1,000
test components, against the oracle and the track's rules. The age, health and
projection rules are calibrated on 800 separate lives. docs/BENCHMARK.md asks for at
least five independently trained models per agent:

    python examples/benchmark_your_agent.py --training-seeds 0 1 2 3 4

--steps 0 skips training and scores only deadline_agent, a hand-written agent that needs
no extra. The command line scores it too:

    python -m maintenance_arena benchmark --track core-continuous-v1 --agent examples.benchmark_your_agent:deadline_agent --output outputs/track-deadline-agent

An agent is any callable agent(observation, info) -> action. The benchmark passes it the
observation and the public info, never the environment. Tune and select on the track's
calibration seeds, never on its test seeds.
"""
import argparse
import importlib.util

import gymnasium as gym

import maintenance_arena  # noqa: F401  (registers the maintenanceArena/* environment IDs)
from maintenance_arena import benchmark, tracks
from maintenance_arena.wrappers import ZeroDeadline


def deadline_agent(observation, info):
    """Order once the deadline coordinate (index 3) reaches -2/13 (0 = WAIT, 1 = REPLACE)."""
    health, wear_rate, rate_valid, deadline = observation
    return int(rate_valid > 0.5 and deadline >= -2 / 13)


def train_sb3_ppo(track, *, steps, seed):
    """Train stock Stable-Baselines3 PPO on the track's environment, created with gymnasium.make."""
    from stable_baselines3 import PPO
    profile = track.resolved_profile()
    env = gym.make(track.environment_id, config=profile["env_config"])
    if track.agent_observation == "zero_deadline":
        env = ZeroDeadline(env)  # train on the input that the track gives to agents
    # Every non-terminal core step lasts one calendar step, so a fixed gamma matches the task's discount.
    model = PPO("MlpPolicy", env, gamma=profile["env_config"]["gamma"], seed=seed, device="cpu")
    model.learn(total_timesteps=steps)
    env.close()
    return lambda observation, info: model.predict(observation, deterministic=True)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default=tracks.DEFAULT_TRACK,
                        help="Core benchmark track: a name from 'maintenance-arena tracks' or a tracks JSON file "
                             f"(default: {tracks.DEFAULT_TRACK})")
    parser.add_argument("--steps", type=int, default=50_000,
                        help="PPO training steps per model (default: 50,000); 0 skips training")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0], metavar="SEED",
                        help="Seeds of independently trained models, one table row each (default: 0)")
    args = parser.parse_args()
    track = tracks.load_track(args.track)
    if track.task != "core":
        parser.error(f"this example trains on the core task; {track.name} is a {track.task} track")
    if args.steps < 0 or len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error("--steps must be nonnegative and the training seeds distinct")
    if args.steps and importlib.util.find_spec("stable_baselines3") is None:
        parser.error("training needs the rl extra (Stable-Baselines3); install it, or use --steps 0 to score only "
                     "deadline_agent")
    agents = {"deadline_agent": deadline_agent}
    if args.steps:
        for seed in args.training_seeds:
            agents[f"sb3_ppo_seed{seed}"] = train_sb3_ppo(track, steps=args.steps, seed=seed)
    result = benchmark.run_track(track, agents)
    print(result.to_markdown())


if __name__ == "__main__":
    main()
