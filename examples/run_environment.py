"""Run every public task, or the one chosen with --profile, through the same policy interface.

Each run uses a conventional policy that reads only the inputs its task declares:

    python examples/run_environment.py
    python examples/run_environment.py --profile paper-resource-lifetime-v1 --policy age
"""
from __future__ import annotations

import argparse
import json

import maintenance_arena
from maintenance_arena.policies import Policy
from maintenance_arena.profiles import all_profiles


def reference_policy(profile_id):
    """Return a suitable conventional reference policy for the profile."""
    return ("age" if "lifetime" in profile_id else "rolling" if "resource" in profile_id
            else "separable_q96" if "gearbox" in profile_id else "projection")


def run(profile_id, policy_name, seed, calendar_steps):
    env = maintenance_arena.make(profile_id, render_mode="ansi")
    try:
        profile = env.unwrapped.arena_profile
        policy = Policy(policy_name, profile, env)
        observation, info = env.reset(seed=seed)
        elapsed = decisions = 0
        raw_profit = 0.0
        terminated = truncated = False
        while elapsed < calendar_steps and not (terminated or truncated):
            observation, reward, terminated, truncated, info = env.step(policy(observation, info))
            elapsed += int(info["duration_calendar_steps"])
            raw_profit += float(info["raw_profit_sum"])
            decisions += 1
        return env.render(), {"profile": profile_id, "policy": policy_name,
                              "seed": seed, "decisions": decisions,
                              "calendar_steps": elapsed, "raw_profit": raw_profit,
                              "terminated": terminated, "truncated": truncated}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=list(all_profiles()), help="Run only this profile (default: every profile)")
    parser.add_argument("--policy", help="Conventional policy for --profile; omitted selects a suitable reference per profile")
    parser.add_argument("--seed", type=int, default=12001, help="Nonnegative scenario seed (default: 12001)")
    parser.add_argument("--calendar-steps", type=int, default=240, help="Calendar-step budget per task (default: 240)")
    args = parser.parse_args()
    if args.seed < 0 or args.calendar_steps < 1:
        parser.error("seed must be nonnegative and calendar-steps positive")
    if args.policy and not args.profile:
        parser.error("--policy requires --profile, because the policy names differ between profiles")
    for profile_id in [args.profile] if args.profile else list(all_profiles()):
        rendered, summary = run(profile_id, args.policy or reference_policy(profile_id), args.seed, args.calendar_steps)
        print(rendered)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
