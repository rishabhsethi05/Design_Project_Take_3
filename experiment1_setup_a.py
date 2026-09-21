"""
experiment1_setup_a.py -- Experiment 1, Setup A (cross-scenario / headline test)

Protocol (matches the professor's spec):
  1. Tune the complete configuration (Q-table + safe_v/tax) on Summer only.
  2. Freeze everything. Change nothing.
  3. Evaluate the frozen configuration on Winter.
  4. Report against TWO references:
       (i)  frozen-Summer-on-Winter vs. the hybrid baseline on Winter
       (ii) frozen-Summer-on-Winter vs. the in-sample Winter-tuned PACE
            (to quantify how much per-scenario tuning was inflating results)

ON REFERENCE (ii) -- WHY THIS SCRIPT COMPUTES TWO VERSIONS OF IT:
The already-published Table IV Winter numbers were generated with
freeze=False (Q-table kept updating during the nominal "evaluation" pass
-- see the harness verification writeup; this is how tuner.py's original
code actually behaved, and it's what produced every number currently in
main.tex). The frozen-Summer transfer in this experiment necessarily
uses freeze=True (that's the entire premise of "freeze everything").
Comparing a freeze=True condition against a freeze=False reference would
confound two different things (scenario transfer AND freeze behavior)
in one number. So this script reports both:
  - INSAMPLE_PUBLISHED: freeze=False, reseed=None -- reproduces Table IV
    exactly (this is "current in-sample Winter-tuned PACE" read literally).
  - INSAMPLE_FROZEN: freeze=True, reseed=None -- same in-sample tuning,
    but with the freeze behavior held constant against the transfer
    condition, isolating the scenario-transfer effect cleanly.
The generalization gap is reported against both, clearly labeled.

SEEDING NOTE: the frozen-Summer transfer's Winter evaluation uses
reseed=<seed> (not reseed=None) -- training happened under SUMMER's
random draws, so continuing that stream would land Winter's evaluation
at an arbitrary, non-canonical point. Reseeding to the same nominal seed
gives a fair, comparable "seed s's Winter realization" against every
other Winter-scenario run tagged with that same seed (hybrid baseline,
in-sample tuning, etc).
"""

import sys
import csv
sys.path.insert(0, ".")

from src.experiment_harness import (
    run_hybrid_once, train_pace_agent, evaluate_pace_agent, run_rung_once,
    average_telemetry, compute_gain
)

SEEDS = [42, 123, 999]
EPOCHS = 15
MULTIPLIER = 5

# Same 22-benchmark tuned profiles used everywhere else in this project.
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


def run_one_benchmark(bench):
    prof = WORKLOAD_PROFILES[bench]
    sv_s, tax_s = prof["Summer"]["safe_v"], prof["Summer"]["tax"]
    sv_w, tax_w = prof["Winter"]["safe_v"], prof["Winter"]["tax"]

    hybrid_runs, transfer_runs, insample_frozen_runs, insample_published_runs = [], [], [], []

    for s in SEEDS:
        # Reference (i): hybrid baseline on Winter, this seed.
        hybrid_runs.append(run_hybrid_once(bench, "Winter", s, multiplier=MULTIPLIER, epochs=1))

        # Frozen-Summer -> Winter transfer (the headline condition).
        summer_agent = train_pace_agent(bench, "Summer", s, sv_s, tax_s, EPOCHS, multiplier=MULTIPLIER)
        transfer_runs.append(
            evaluate_pace_agent(bench, "Winter", summer_agent, multiplier=MULTIPLIER, freeze=True, reseed=s)
        )

        # Reference (ii-a): in-sample Winter tuning, freeze=True (fair, matched-freeze comparison).
        winter_agent = train_pace_agent(bench, "Winter", s, sv_w, tax_w, EPOCHS, multiplier=MULTIPLIER)
        insample_frozen_runs.append(
            evaluate_pace_agent(bench, "Winter", winter_agent, multiplier=MULTIPLIER, freeze=True, reseed=None)
        )

        # Reference (ii-b): in-sample Winter tuning, freeze=False ("as published," reproduces Table IV).
        insample_published_runs.append(
            run_rung_once(bench, "Winter", s, sv_w, tax_w, EPOCHS, multiplier=MULTIPLIER)
        )

    avg_hybrid = average_telemetry(hybrid_runs)
    avg_transfer = average_telemetry(transfer_runs)
    avg_insample_frozen = average_telemetry(insample_frozen_runs)
    avg_insample_published = average_telemetry(insample_published_runs)

    if not (avg_hybrid and avg_transfer and avg_insample_frozen and avg_insample_published):
        return None

    gain_transfer = compute_gain(avg_hybrid.total_cycles, avg_transfer.total_cycles)
    gain_insample_frozen = compute_gain(avg_hybrid.total_cycles, avg_insample_frozen.total_cycles)
    gain_insample_published = compute_gain(avg_hybrid.total_cycles, avg_insample_published.total_cycles)

    gap_vs_frozen = (gain_insample_frozen - gain_transfer) if (gain_insample_frozen is not None and gain_transfer is not None) else None
    gap_vs_published = (gain_insample_published - gain_transfer) if (gain_insample_published is not None and gain_transfer is not None) else None

    return {
        "benchmark": bench.replace(".c", ""),
        "hybrid_winter_cycles": avg_hybrid.total_cycles,
        "transfer_gain": gain_transfer,
        "insample_frozen_gain": gain_insample_frozen,
        "insample_published_gain": gain_insample_published,
        "generalization_gap_vs_frozen_ref": gap_vs_frozen,
        "generalization_gap_vs_published_ref": gap_vs_published,
    }


if __name__ == "__main__":
    print(f"{'='*100}\nEXPERIMENT 1 SETUP A -- Cross-scenario generalization (Summer-tuned, frozen, evaluated on Winter)\n{'='*100}")
    print(f"Seeds: {SEEDS} | Epochs: {EPOCHS} | Multiplier: {MULTIPLIER}\n")

    rows = []
    for bench in WORKLOAD_PROFILES:
        print(f"[+] {bench} ...", flush=True)
        result = run_one_benchmark(bench)
        if result is None:
            print(f"    SKIPPED (invalid run -- one condition failed to produce valid cycles)")
            continue
        rows.append(result)
        print(f"    frozen-transfer: {result['transfer_gain']:7.2f}% | "
              f"in-sample (frozen): {result['insample_frozen_gain']:7.2f}% | "
              f"in-sample (published): {result['insample_published_gain']:7.2f}% | "
              f"gap (vs published): {result['generalization_gap_vs_published_ref']:7.2f} pts")

    # ---- Summary table ----
    print(f"\n{'='*100}\nSUMMARY\n{'='*100}")
    print(f"{'Benchmark':<16}| {'Transfer':>10} | {'InSample(F)':>12} | {'InSample(Pub)':>14} | {'Gap(F)':>8} | {'Gap(Pub)':>9}")
    print("-" * 100)
    for r in rows:
        print(f"{r['benchmark']:<16}| {r['transfer_gain']:>9.2f}% | {r['insample_frozen_gain']:>11.2f}% | "
              f"{r['insample_published_gain']:>13.2f}% | {r['generalization_gap_vs_frozen_ref']:>7.2f} | "
              f"{r['generalization_gap_vs_published_ref']:>8.2f}")

    # ---- Aggregate stats ----
    import numpy as np
    transfer_vals = [r["transfer_gain"] for r in rows]
    insample_pub_vals = [r["insample_published_gain"] for r in rows]
    gaps_pub = [r["generalization_gap_vs_published_ref"] for r in rows]

    print(f"\n{'='*100}\nAGGREGATE (n={len(rows)})\n{'='*100}")
    print(f"Frozen-transfer      : mean={np.mean(transfer_vals):.2f}%  median={np.median(transfer_vals):.2f}%  "
          f"win-rate={sum(1 for v in transfer_vals if v>0)}/{len(transfer_vals)}")
    print(f"In-sample (published): mean={np.mean(insample_pub_vals):.2f}%  median={np.median(insample_pub_vals):.2f}%  "
          f"win-rate={sum(1 for v in insample_pub_vals if v>0)}/{len(insample_pub_vals)}")
    print(f"Generalization gap (published ref): mean={np.mean(gaps_pub):.2f} pts  median={np.median(gaps_pub):.2f} pts")

    # ---- CSV export ----
    with open("experiment1_setup_a_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n[*] Saved experiment1_setup_a_results.csv")