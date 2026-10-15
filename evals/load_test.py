"""Load test: concurrent /query traffic -> p50/p99 table.

    python3 evals/load_test.py [--port 8931] [--concurrency 1,4,16] [--n 40]

Spawns uvicorn serving the mock backend (RAG_PROFILE=load: cache off,
keyless), indexes the sample docs, then hammers /query with the golden
questions at each concurrency level and prints a latency table. The
per-stage breakdown is server-side (pipeline timers), the total is
client wall-clock so queueing under concurrency shows up.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx  # noqa: E402

PORT_DEFAULT = 8931


def wait_ready(base, timeout=90):
    # uvicorn + model-less import is fast, but poll anyway so the
    # script never races the server into a connection error
    deadline = time.time() + timeout
    with httpx.Client(trust_env=False) as c:  # proxy env breaks httpx's
                                                   # url pattern parser here
        while time.time() < deadline:
            try:
                if c.get(f"{base}/health").status_code == 200:
                    return
            except httpx.ConnectError:
                time.sleep(0.3)
    raise RuntimeError("server never came up")


def pct(data, q):
    if not data:
        return float("nan")
    s = sorted(data)
    i = min(int(q / 100 * len(s)), len(s) - 1)
    return s[i]


def fire(args, base, concurrency):
    # one shared client: connection pool amortizes handshake cost,
    # per-request wall clock captures queueing at high concurrency
    client = httpx.Client(trust_env=False, timeout=120, limits=httpx.Limits(
        max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4))
    questions = [json.loads(l) for l in open("evals/golden.jsonl")
                 if l.strip()]
    results = []

    def one(i):
        q = questions[i % len(questions)]
        t0 = time.perf_counter()
        try:
            r = client.post(f"{base}/query",
                            json={"text": q["question"],
                                  "filters": q.get("filters"), "top_k": 3})
            r.raise_for_status()
            body = r.json()
            return {"ok": True,
                    "client_ms": (time.perf_counter() - t0) * 1000,
                    "server_ms": body.get("latency_ms", 0),
                    "breakdown": body.get("breakdown", {})}
        except Exception:
            # keep going on failures — error count is part of the report
            return {"ok": False,
                    "client_ms": (time.perf_counter() - t0) * 1000,
                    "server_ms": 0, "breakdown": {}}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(args.n)))
    client.close()
    return results


def print_table(levels):
    # levels: {concurrency: [results]}
    def row(c, res):
        ok = [r for r in res if r["ok"]]
        cm = [r["client_ms"] for r in ok]
        sm = [r["server_ms"] for r in ok]
        return (c, len(ok), len(res) - len(ok),
                pct(cm, 50), pct(cm, 99), max(cm) if cm else float("nan"),
                pct(sm, 50), pct(sm, 99))

    print(f"{'concurrency':>11} {'n':>4} {'err':>4} "
          f"{'client p50':>10} {'client p99':>10} {'client max':>10} "
          f"{'server p50':>10} {'server p99':>10}  (ms)")
    for c, res in sorted(levels.items()):
        _, n, err, cp50, cp99, cmax, sp50, sp99 = row(c, res)
        print(f"{c:>11} {n:>4} {err:>4} "
              f"{cp50:>10.1f} {cp99:>10.1f} {cmax:>10.1f} "
              f"{sp50:>10.1f} {sp99:>10.1f}")
    # per-stage server timings at the heaviest level — shows which stage
    # degrades under concurrency (usually retrieve on the GIL)
    heavy = levels[max(levels)]
    ok = [r for r in heavy if r["ok"] and r["breakdown"]]
    if ok:
        stages = sorted({k for r in ok for k in r["breakdown"]})
        print(f"\nper-stage server ms @ concurrency {max(levels)} "
              f"(n={len(ok)}):")
        print(f"{'stage':>10} {'p50':>8} {'p99':>8} {'max':>8}")
        for s in stages:
            vals = [r["breakdown"][s] for r in ok if s in r["breakdown"]]
            print(f"{s:>10} {pct(vals, 50):>8.1f} {pct(vals, 99):>8.1f} "
                  f"{max(vals):>8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT_DEFAULT)
    ap.add_argument("--concurrency", default="1,4,16")
    ap.add_argument("--n", type=int, default=40,
                    help="requests per concurrency level")
    args = ap.parse_args()
    levels = [int(x) for x in args.concurrency.split(",")]
    base = f"http://127.0.0.1:{args.port}"

    env = dict(os.environ)
    env["RAG_PROFILE"] = "load"  # cache off, mock generator — see config.load.yaml
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "service:app",
         "--host", "127.0.0.1", "--port", str(args.port),
         "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(base)
        # index the corpus through the public endpoint, like a real deploy
        docs = json.load(open("data/sample_docs.json"))
        r = httpx.post(f"{base}/index",
                       json=[{"text": d["text"],
                              "metadata": d.get("metadata", {})}
                             for d in docs], timeout=120,
                       trust_env=False)
        r.raise_for_status()
        print(f"indexed {r.json()['chunks_indexed']} chunks")
        results = {c: fire(args, base, c) for c in levels}
        print_table(results)
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
