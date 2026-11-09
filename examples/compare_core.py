"""Compare the core policies on paired scenarios through the public command line.

    python examples/compare_core.py --seed 12001 --output outputs/core-comparison

This needs the plots extra. --checkpoint adds a trained ppo policy, which also needs the
rl extra. The outputs record every per-unit metric and the source identity.
"""
import argparse
from maintenance_arena.cli import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    policies = ["reactive", "age", "health", "projection", "immediate"]
    command = ["evaluate", "--profile", "paper-core-continuous-v1", "--seed", str(args.seed),
               "--output", args.output, "--episodes", str(args.episodes), "--calendar-steps", "240", "--plot"]
    if args.checkpoint:
        policies.append("ppo")
        command += ["--checkpoint", args.checkpoint]
    raise SystemExit(main(command + ["--policy", *policies]))
