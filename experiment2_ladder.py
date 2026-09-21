"""
experiment2_ladder.py -- Experiment 2 (attribution ladder)

Four configurations, IDENTICAL seed set across all four, run on both
scenarios, per benchmark:

  R1  Hybrid heuristic alone                          -- true baseline
  R2  Hybrid + deterministic safety reflexes            -- what the shield adds
  R3  Hybrid + Q-agent, safety reflexes disabled         -- can the agent learn safety unaided?
  R4  Full PACE (Hybrid + shield + Q-agent)              -- what the agent adds on top of the shield

The decisive number is R4 - R2: the Q-agent's genuine incremental
contribution, isolated from what the hard-coded safety reflexes already
provide. This is the comparison the current paper (main.tex) cannot
produce, since it never separates the reflex layer from the agent.

PRE-REGISTERED PREDICTION (written before running, per the professor's
Phase 1 item 5): We predict R2 will recover most of R1's crash-driven
losses on benchmarks where crashes dominate the baseline's cost (i.e.
R2's crash_count will drop sharply vs R1), since the reflexes are a
blunt but effective brownout guard. We predict R3 (agent without
reflexes) will show WORSE crash behavior than R2 on at least the
already-identified fragile benchmarks (bs, cnt, cubic, fibcall,
recursion, select) -- i.e. the agent alone does not reliably learn the
safety behavior the hard-coded reflexes provide "for free." We predict
R4-R2 will be small and sometimes negative on those same fragile
benchmarks (the agent adds little or hurts once the shield already
handles safety), and larger and positive on the compute-heavy trio
(aes, sha256, fir), where there is more genuine "gray area" execution
length for the agent to optimize within.

SEED SET: uses a small, dedicated held-out set (distinct from the
original tuning seeds AND from Setup B's held-out pool, to keep the
three experiments' seed provenance unambiguous if anyone audits this
later) -- 5 seeds, balancing statistical value against runtime, since
each benchmark now requires 4 rungs x 2 scenarios x 5 seeds = 40 runs,
several of which involve full training.
"""

import sys
import csv
import numpy as np
sys.path.insert(0, ".")

from src.experiment_harness import (
    run_hybrid_once, train_pace_agent, evaluate_pace_agent, run_rung_once,
    average_telemetry, RunTelemetry
)
from src.experiment_harness import compute_gain

LADDER_SEEDS = [201, 202, 203, 204, 205]  # dedicated to this experiment -- see docstring
EPOCHS = 15
MULTIPLIER = 5
SCENARIOS = ["Summer", "Winter"]

WORKLOAD_PROFILES = {
    "aes.c":           {"Summer": {"safe_v": 2.00, "tax": 0.50}, "Winter": {"safe_v": 2.00, "tax": 1.00}},
    "bs.c":            {"Summer": {"safe_v": 1.95, "tax": 0.50}, "Winter": {"safe_v": 2.00, "tax": 0.00}},
    "cnt.c":           {"Summer": {"safe_v": 2.00, "tax": 0.00}, "Winter": {"safe_v": 2.00, "tax": 0.50}},
    "crc.c":           {"Summer": {"safe_v": 2.00, "tax": 1.00}, "Winter": {"safe_v": 1.90, "tax": 0.25}},
    "dct.c":           {"Summer": {"safe_v": 1.96, "tax": 0.05}, "Winter": {"safe_v": 1.90, "tax": 0.50}},
    "dijkstra.c":      {"Summer": {"safe_v": 2.00, "tax": 7.50}, "Winter": {"safe_v": 2.00, "tax": 0.00}},
    "fir.c":           {"Summer": {"safe_v": 2.00, "tax": 7.50}, "Winter": {"safe_v": 2.00, "tax": 0.00}},
    "huffman.c":       {"Summer": {"safe_v": 2.00, "tax": 0.50}, "Winter": {"safe_v": 2.00, "tax": 0.00}},
    "mergesort.c":     {"Summer": {"safe_v": 2.00, "tax": 0.50}, "Winter": {"safe_v": 2.00, "tax": 0.00}},
    "prime.c":         {"Summer": {"safe_v": 2.00, "tax": 1.00}, "Winter": {"safe_v": 2.00, "tax": 0.50}},
    "quicksort.c":     {"Summer": {"safe_v": 2.00, "tax": 0.50}, "Winter": {"safe_v": 2.00, "tax": 0.50}},
    "select.c":        {"Summer": {"safe_v": 1.90, "tax": 4.75}, "Winter": {"safe_v": 1.90, "tax": 0.00}},
    "sha256.c":        {"Summer": {"safe_v": 2.00, "tax": 7.50}, "Winter": {"safe_v": 2.00, "tax": 0.00}},
    "strstr.c":        {"Summer": {"safe_v": 1.90, "tax": 0.25}, "Winter": {"safe_v": 2.00, "tax": 0.00}},
    "wikisort.c":      {"Summer": {"safe_v": 1.90, "tax": 0.50}, "Winter": {"safe_v": 1.95, "tax": 0.75}},
    "bubblesort.c":    {"Summer": {"safe_v": 2.00, "tax": 1.00}, "Winter": {"safe_v": 1.90, "tax": 0.50}},
    "cubic.c":         {"Summer": {"safe_v": 2.00, "tax": 0.00}, "Winter": {"safe_v": 1.90, "tax": 0.25}},
    "fibcall.c":       {"Summer": {"safe_v": 1.90, "tax": 0.25}, "Winter": {"safe_v": 2.00, "tax": 0.00}},
    "hash.c":          {"Summer": {"safe_v": 1.90, "tax": 0.25}, "Winter": {"safe_v": 2.00, "tax": 0.50}},
    "insertsort.c":    {"Summer": {"safe_v": 1.90, "tax": 0.00}, "Winter": {"safe_v": 2.00, "tax": 2.50}},
    "recursion.c":     {"Summer": {"safe_v": 2.00, "tax": 0.50}, "Winter": {"safe_v": 1.90, "tax": 0.00}},
    "stringsearch1.c": {"Summer": {"safe_v": 2.00, "tax": 0.00}, "Winter": {"safe_v": 2.00, "tax": 0.00}},
}

# Category membership, matching main.tex Table VI exactly (for later
# category-level aggregation once we get to the writing phase).
CATEGORIES = {
    "Sorting/Search": ["bubblesort", "insertsort", "quicksort", "mergesort", "wikisort",
                        "strstr", "stringsearch1", "select", "bs"],
    "Control-flow/Numeric": ["cnt", "crc", "prime", "cubic", "fibcall", "recursion",
                              "hash", "dct", "dijkstra", "huffman"],
    "Compute-heavy": ["aes", "sha256", "fir"],
}

# Edit for a faster test pass before the full sweep, e.g.:
#   BENCHMARKS_SUBSET = ["aes.c", "crc.c", "fibcall.c"]
BENCHMARKS_SUBSET = list(WORKLOAD_PROFILES.keys())  # full 22 by default


def run_rung(bench, scenario, rung, safe_v, tax):
    telemetries = []
    for s in LADDER_SEEDS:
        if rung == "R1":
            t = run_hybrid_once(bench, scenario, s, multiplier=MULTIPLIER, epochs=1)
        elif rung == "R2":
            t = run_rung_once(bench, scenario, s, safe_v, tax, EPOCHS, multiplier=MULTIPLIER,
                               use_reflexes=True, use_agent=False)
        elif rung == "R3":
            t = run_rung_once(bench, scenario, s, safe_v, tax, EPOCHS, multiplier=MULTIPLIER,
                               use_reflexes=False, use_agent=True)
        elif rung == "R4":
            t = run_rung_once(bench, scenario, s, safe_v, tax, EPOCHS, multiplier=MULTIPLIER,
                               use_reflexes=True, use_agent=True)
        else:
            raise ValueError(rung)
        telemetries.append(t)
    return average_telemetry(telemetries)


def run_one_benchmark_scenario(bench, scenario):
    safe_v, tax = WORKLOAD_PROFILES[bench][scenario]["safe_v"], WORKLOAD_PROFILES[bench][scenario]["tax"]
    rungs = {}
    for rung in ["R1", "R2", "R3", "R4"]:
        rungs[rung] = run_rung(bench, scenario, rung, safe_v, tax)
    if any(rungs[r] is None for r in rungs):
        return None

    r1_cycles = rungs["R1"].total_cycles
    gains = {r: compute_gain(r1_cycles, rungs[r].total_cycles) for r in ["R2", "R3", "R4"]}
    r4_minus_r2 = (gains["R4"] - gains["R2"]) if (gains["R4"] is not None and gains["R2"] is not None) else None

    row = {"benchmark": bench.replace(".c", ""), "scenario": scenario}
    for r in ["R1", "R2", "R3", "R4"]:
        t: RunTelemetry = rungs[r]
        row[f"{r}_cycles"] = t.total_cycles
        row[f"{r}_crash_count"] = t.crash_count
        row[f"{r}_wasted_cycles"] = t.wasted_cycles
        row[f"{r}_checkpoint_count"] = t.checkpoint_count
        row[f"{r}_active_cycles"] = t.active_cycles
        row[f"{r}_sleep_cycles"] = t.sleep_cycles
    row["R2_gain_vs_R1"] = gains["R2"]
    row["R3_gain_vs_R1"] = gains["R3"]
    row["R4_gain_vs_R1"] = gains["R4"]
    row["R4_minus_R2"] = r4_minus_r2
    return row


if __name__ == "__main__":
    print(f"{'='*100}\nEXPERIMENT 2 -- Attribution Ladder (R1-R4)\n{'='*100}")
    print(f"Seeds: {LADDER_SEEDS} | Epochs: {EPOCHS} | Multiplier: {MULTIPLIER}")
    print(f"Benchmarks: {len(BENCHMARKS_SUBSET)} | Scenarios: {SCENARIOS}\n")

    rows = []
    for bench in BENCHMARKS_SUBSET:
        for scenario in SCENARIOS:
            print(f"[+] {bench} / {scenario} ...", flush=True)
            row = run_one_benchmark_scenario(bench, scenario)
            if row is None:
                print(f"    SKIPPED (a rung failed to produce valid telemetry)")
                continue
            rows.append(row)
            print(f"    R2 vs R1: {row['R2_gain_vs_R1']:7.2f}%  (crashes {row['R1_crash_count']}->{row['R2_crash_count']})  |  "
                  f"R3 vs R1: {row['R3_gain_vs_R1']:7.2f}%  (crashes {row['R3_crash_count']})  |  "
                  f"R4 vs R1: {row['R4_gain_vs_R1']:7.2f}%  (crashes {row['R4_crash_count']})  |  "
                  f"R4-R2: {row['R4_minus_R2']:7.2f} pts")

    print(f"\n{'='*100}\nSUMMARY (the decisive column is R4-R2)\n{'='*100}")
    print(f"{'Benchmark':<16}| {'Scen':<7}| {'R2%':>8} | {'R3%':>8} | {'R4%':>8} | {'R4-R2':>8} | R1/R2/R3/R4 crashes")
    print("-" * 100)
    for r in rows:
        print(f"{r['benchmark']:<16}| {r['scenario']:<7}| {r['R2_gain_vs_R1']:>7.2f}% | {r['R3_gain_vs_R1']:>7.2f}% | "
              f"{r['R4_gain_vs_R1']:>7.2f}% | {r['R4_minus_R2']:>7.2f} | "
              f"{r['R1_crash_count']}/{r['R2_crash_count']}/{r['R3_crash_count']}/{r['R4_crash_count']}")

    # ---- Category-level R4-R2 aggregation (for the professor's requested table) ----
    print(f"\n{'='*100}\nR4-R2 BY CATEGORY (mean, per scenario)\n{'='*100}")
    for scenario in SCENARIOS:
        print(f"\n{scenario}:")
        for cat, members in CATEGORIES.items():
            deltas = [r["R4_minus_R2"] for r in rows if r["scenario"] == scenario and r["benchmark"] in members]
            if deltas:
                print(f"  {cat:<24}: mean R4-R2 = {np.mean(deltas):7.2f} pts  (n={len(deltas)})")

    with open("experiment2_ladder_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n[*] Saved experiment2_ladder_results.csv")