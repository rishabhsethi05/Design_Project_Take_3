"""
experiment1_setup_b.py -- Experiment 1, Setup B (cross-trace robustness)
FIXED: previously silently dropped failed (invalid) evaluations from the
mean/std (survivorship bias -- worst outcomes excluded instead of
counted). Now every dropped evaluation is tallied as an explicit
failure, and failure_rate is reported as its own statistic per
(benchmark, scenario) alongside mean/std computed over the survivors.
"""

import sys
import csv
import numpy as np
sys.path.insert(0, ".")

from src.experiment_harness import (
    run_hybrid_once, train_pace_agent, evaluate_pace_agent, compute_gain
)

TRAIN_SEEDS = [42, 123, 999]
HELD_OUT_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
EPOCHS = 15
MULTIPLIER = 5

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

BENCHMARKS_SUBSET = list(WORKLOAD_PROFILES.keys())
SCENARIOS = ["Summer", "Winter"]


def run_one(bench, scenario):
    prof = WORKLOAD_PROFILES[bench][scenario]
    safe_v, tax = prof["safe_v"], prof["tax"]

    hybrid_by_seed = {hs: run_hybrid_once(bench, scenario, hs, multiplier=MULTIPLIER, epochs=1)
                       for hs in HELD_OUT_SEEDS}

    all_gains = []
    n_failed = 0
    per_train_seed_gains = {}

    for train_seed in TRAIN_SEEDS:
        agent = train_pace_agent(bench, scenario, train_seed, safe_v, tax, EPOCHS, multiplier=MULTIPLIER)
        seed_gains = []
        for hs in HELD_OUT_SEEDS:
            pace_telem = evaluate_pace_agent(bench, scenario, agent, multiplier=MULTIPLIER,
                                              freeze=True, reseed=hs)
            hybrid_telem = hybrid_by_seed[hs]
            gain = compute_gain(hybrid_telem.total_cycles, pace_telem.total_cycles)
            if gain is not None:
                seed_gains.append(gain)
                all_gains.append(gain)
            else:
                n_failed += 1  # FIX: explicitly counted, not silently dropped
        per_train_seed_gains[train_seed] = seed_gains

    n_expected = len(TRAIN_SEEDS) * len(HELD_OUT_SEEDS)
    assert len(all_gains) + n_failed == n_expected, "accounting mismatch -- every evaluation must be either a gain or a counted failure"

    if not all_gains:
        return {
            "benchmark": bench.replace(".c", ""), "scenario": scenario,
            "n_valid": 0, "n_expected": n_expected, "n_failed": n_failed,
            "failure_rate": 1.0,
            "mean_gain": None, "std_gain": None, "min_gain": None, "max_gain": None,
        }

    return {
        "benchmark": bench.replace(".c", ""),
        "scenario": scenario,
        "n_valid": len(all_gains),
        "n_expected": n_expected,
        "n_failed": n_failed,
        "failure_rate": n_failed / n_expected,
        "mean_gain": float(np.mean(all_gains)),
        "std_gain": float(np.std(all_gains)),
        "min_gain": float(np.min(all_gains)),
        "max_gain": float(np.max(all_gains)),
    }


if __name__ == "__main__":
    print(f"{'='*100}\nEXPERIMENT 1 SETUP B -- Cross-trace robustness (held-out seeds: {HELD_OUT_SEEDS})\n{'='*100}")
    print(f"Train seeds: {TRAIN_SEEDS} | Epochs: {EPOCHS} | Multiplier: {MULTIPLIER}")
    print(f"Benchmarks: {len(BENCHMARKS_SUBSET)} | Scenarios: {SCENARIOS}\n")

    rows = []
    for bench in BENCHMARKS_SUBSET:
        for scenario in SCENARIOS:
            print(f"[+] {bench} / {scenario} ...", flush=True)
            result = run_one(bench, scenario)
            rows.append(result)
            if result["mean_gain"] is None:
                print(f"    ALL {result['n_expected']} EVALUATIONS FAILED (failure_rate=100%)")
                continue
            fail_flag = "" if result["n_failed"] == 0 else \
                f"  [!] failure_rate={result['failure_rate']*100:.1f}% ({result['n_failed']}/{result['n_expected']})"
            print(f"    mean={result['mean_gain']:7.2f}%  std={result['std_gain']:6.2f}  "
                  f"range=[{result['min_gain']:.2f}, {result['max_gain']:.2f}]{fail_flag}")

    print(f"\n{'='*100}\nSUMMARY\n{'='*100}")
    print(f"{'Benchmark':<16}| {'Scenario':<8}| {'Mean':>8} | {'Std':>7} | {'Min':>9} | {'Max':>9} | {'Fail%':>7} | N")
    print("-" * 100)
    for r in rows:
        if r["mean_gain"] is None:
            print(f"{r['benchmark']:<16}| {r['scenario']:<8}| {'--':>8} | {'--':>7} | {'--':>9} | {'--':>9} | "
                  f"{'100.0%':>7} | 0/{r['n_expected']}")
            continue
        print(f"{r['benchmark']:<16}| {r['scenario']:<8}| {r['mean_gain']:>7.2f}% | {r['std_gain']:>6.2f} | "
              f"{r['min_gain']:>8.2f}% | {r['max_gain']:>8.2f}% | {r['failure_rate']*100:>6.1f}% | "
              f"{r['n_valid']}/{r['n_expected']}")

    with open("experiment1_setup_b_results.csv", "w", newline="") as f:
        fieldnames = ["benchmark", "scenario", "n_valid", "n_expected", "n_failed", "failure_rate",
                      "mean_gain", "std_gain", "min_gain", "max_gain"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print("\n[*] Saved experiment1_setup_b_results.csv")