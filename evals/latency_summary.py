"""Latency summary: per-stage p50/p99 from the JSONL query log.

    python3 evals/latency_summary.py                  # logs/latency.jsonl
    python3 evals/latency_summary.py logs/other.jsonl

Each line of the log is one query, written by pipeline._record_latency:
ts, question, backend, top_k, total_ms, stages_ms, index_version.
This script aggregates the stage timings into a table — p50 tells you
the typical cost, p99 tells you what the slowest 1% look like, and the
stage with the fattest p50 is the one worth optimizing first.
"""
import json
import statistics
import sys


def _pct(vals, q):
    # percentile without numpy — sorted + linear interpolation
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = (len(vals) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def summarize(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    stages = sorted({s for r in rows for s in r.get("stages_ms", {})})
    totals = [r["total_ms"] for r in rows]
    table = {"_n": len(rows)}
    for s in stages:
        vals = [r["stages_ms"].get(s, 0.0) for r in rows]
        table[s] = {"mean": statistics.fmean(vals),
                    "p50": _pct(vals, 50), "p99": _pct(vals, 99)}
    table["_total"] = {"mean": statistics.fmean(totals),
                       "p50": _pct(totals, 50), "p99": _pct(totals, 99)}
    backends = sorted({r.get("backend", "?") for r in rows})
    return table, backends


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/latency.jsonl"
    try:
        table, backends = summarize(path)
    except FileNotFoundError:
        print(f"no log at {path} — run evals/latency_probe.py first")
        sys.exit(1)
    n = table.pop("_n")
    print(f"{path}: {n} queries"
          + (f" (backend: {', '.join(backends)})" if backends else ""))
    print(f"{'stage':12s} {'mean_ms':>9s} {'p50_ms':>9s} {'p99_ms':>9s}")
    for stage, agg in table.items():
        print(f"{stage:12s} {agg['mean']:9.1f} {agg['p50']:9.1f} "
              f"{agg['p99']:9.1f}")


if __name__ == "__main__":
    main()
