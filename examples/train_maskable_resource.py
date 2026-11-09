"""Train MaskablePPO with VecNormalize on Resource-v0 and score it on the resource track.

With the rl extra (PyTorch, Stable-Baselines3, sb3-contrib), run from the repository root:

    python examples/train_maskable_resource.py

This trains sb3-contrib's MaskablePPO for 20,480 steps on the CPU, with the PPO settings
of paper-resource-v1 (eight environments, 256 steps per environment and update). It
scores the frozen model on the track resource-v1, with 200 fixed 720-hour test scenarios,
against the 48-hour rolling-horizon planner and the profile's rules. The run takes a few
minutes, and its short budget only checks the workflow.

The resource tasks need two additions to a stock setup. Actions are masked: MaskablePPO
reads the mask through the environment's action_masks() in training, and the agent passes
info["action_mask"] to predict in scoring. Observations are normalized, because the raw
fields differ widely in magnitude and have no upper bound. VecNormalize learns running
statistics in training, and the agent applies them frozen to every observation.
"""
import argparse
import importlib.util

from maintenance_arena import benchmark, tracks


def train(track, *, steps, seed):
    """Train MaskablePPO with VecNormalize on the track's environment and return a frozen agent."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    recipe = track.resolved_profile()["training"]
    hyperparameters = dict(recipe["hyperparameters"])
    # make_track builds Resource-v0 with the track's horizon and calendar budget.
    venv = DummyVecEnv([lambda: tracks.make_track(track)] * int(recipe["n_envs"]))
    venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
                        gamma=hyperparameters["gamma"])
    model = MaskablePPO("MlpPolicy", venv, seed=seed, device="cpu", **hyperparameters)
    model.learn(total_timesteps=steps)
    venv.training = False  # freeze the observation statistics for scoring

    def agent(observation, info):
        action, _ = model.predict(venv.normalize_obs(observation), action_masks=info["action_mask"],
                                  deterministic=True)
        return action

    return agent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default="resource-v1",
                        help="Resource benchmark track: a name from 'maintenance-arena tracks' or a tracks JSON file "
                             "(default: resource-v1)")
    parser.add_argument("--steps", type=int, default=20_480, help="Training steps (default: 20,480)")
    parser.add_argument("--seed", type=int, default=0, help="Training seed (default: 0)")
    args = parser.parse_args()
    track = tracks.load_track(args.track)
    if track.task != "resource":
        parser.error(f"this example trains on a resource task; {track.name} is a {track.task} track")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if importlib.util.find_spec("sb3_contrib") is None:
        parser.error("this example needs the rl extra (PyTorch, Stable-Baselines3 and sb3-contrib)")
    agent = train(track, steps=args.steps, seed=args.seed)
    result = benchmark.run_track(track, {f"maskable_ppo_seed{args.seed}": agent})
    print(result.to_markdown())


if __name__ == "__main__":
    main()
