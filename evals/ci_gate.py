#!/usr/bin/env python3
"""CI gate: fail when any backend's recall@3 drops below the committed baseline.

run_eval.py rewrites evals/last-eval-snapshot.json on every run, so CI copies
the checked-in file aside as the baseline before running evals, then runs:

    python evals/ci_gate.py --baseline /tmp/eval-baseline.json \
        --current evals/last-eval-snapshot.json

exit 0 when every backend holds or improves, 1 on any regression.
"""
import argparse
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def recall(snapshot):
    return {name: row["recall@3"]
            for name, row in snapshot["per_backend"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    args = ap.parse_args()

    try:
        base, cur = recall(load(args.baseline)), recall(load(args.current))
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"gate error: can't read snapshots ({e})")
        return 2

    # evals on the checked-in golden set are the contract — the gate
    # compares like with like, so both files must name the same backends
    if set(base) != set(cur):
        print(f"gate error: backends differ: {sorted(base)} vs {sorted(cur)}")
        return 2

    print(f"{'backend':10s} {'baseline':>8s} {'current':>7s} {'delta':>6s}")
    fails = 0
    for name in sorted(base):
        d = cur[name] - base[name]
        print(f"{name:10s} {base[name]:>8d} {cur[name]:>7d} {d:>+6d}")
        if d < 0:
            print(f"  REGRESSION: {name} recall@3 fell {base[name]} -> {cur[name]}")
            fails += 1
    if fails:
        print(f"gate failed: {fails} backend(s) regressed")
        return 1
    print("gate passed: recall@3 held on every backend")
    return 0


if __name__ == "__main__":
    sys.exit(main())
