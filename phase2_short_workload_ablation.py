"""
phase2_short_workload_ablation.py -- Phase 2, item 6

Tests the hypothesis currently stated (as an untested hypothesis) in
main.tex Section IV-D: that short benchmarks underperform because fixed
checkpoint overheads (the spam-tax penalty, the hardcoded safety
reflexes) have too little execution length to amortize against.

METHOD: run full PACE (R4, matching the exact protocol that produced
Table IV -- train then evaluate with freeze=False, reseed=None, same as
tuner.py) on all 22 benchmarks under Winter, and compute two normalized
metrics per benchmark:

  checkpoint_density = checkpoint_count / hybrid_baseline_cycles
      "How many checkpoints does PACE take per unit of baseline work?"
      A fixed per-checkpoint cost matters more when this is high.

  waste_fraction = wasted_cycles / pace_total_cycles
      "What fraction of PACE's own execution was pure crash-recompute
      loss?" Directly measures whether crashes (not checkpoint overhead
      itself) are what's actually killing these benchmarks.

Then splits benchmarks into the ALREADY-IDENTIFIED both-scenarios-negative
group (bs, cnt, cubic, fibcall, recursion, select -- from main.tex
Section IV-D) vs. everything else, and compares the two groups on both
metrics. If the amortization hypothesis is right, the short/fragile group
should show markedly higher checkpoint_density. If waste_fraction is
instead what's elevated (and checkpoint_density is NOT), that points to
crash-loop dynamics (Winter's SPF stochasticity) as the real driver
instead of, or in addition to, fixed-overhead amortization -- a
DIFFERENT explanation than the one currently in the paper.

Also reports the correlation between hybrid_baseline_cycles (a length
proxy) and each metric across all 22 benchmarks, not just the two-group
split, since a real amortization effect should show up as a continuous
trend, not just a group difference.
"""

import sys
import csv
import numpy as np
sys.path.insert(0, ".")

from src.experiment_harness import run_hybrid_once, train_pace_agent, evaluate_pace_agent, average_telemetry

TRAIN_SEEDS = [42, 123, 999]
EPOCHS = 15
MULTIPLIER = 5
SCENARIO = "Winter"  # this is where the amortization hypothesis was raised in the paper

WORKLOAD_PROFILES = {
    "aes.c":           {"safe_v": 2.00, "tax": 1.00},
    "bs.c":            {"safe_v": 2.00, "tax": 0.00},
    "cnt.c":           {"safe_v": 2.00, "tax": 0.50},
    "crc.c":           {"safe_v": 1.90, "tax": 0.25},
    "dct.c":           {"safe_v": 1.90, "tax": 0.50},
    "dijkstra.c":      {"safe_v": 2.00, "tax": 0.00},
    "fir.c":           {"safe_v": 2.00, "tax": 0.00},
    "huffman.c":       {"safe_v": 2.00, "tax": 0.00},
    "mergesort.c":     {"safe_v": 2.00, "tax": 0.00},
    "prime.c":         {"safe_v": 2.00, "tax": 0.50},
    "quicksort.c":     {"safe_v": 2.00, "tax": 0.50},
    "select.c":        {"safe_v": 1.90, "tax": 0.00},
    "sha256.c":        {"safe_v": 2.00, "tax": 0.00},
    "strstr.c":        {"safe_v": 2.00, "tax": 0.00},
    "wikisort.c":      {"safe_v": 1.95, "tax": 0.75},
    "bubblesort.c":    {"safe_v": 1.90, "tax": 0.50},
    "cubic.c":         {"safe_v": 1.90, "tax": 0.25},
    "fibcall.c":       {"safe_v": 2.00, "tax": 0.00},
    "hash.c":          {"safe_v": 2.00, "tax": 0.50},
    "insertsort.c":    {"safe_v": 2.00, "tax": 2.50},
    "recursion.c":     {"safe_v": 1.90, "tax": 0.00},
    "stringsearch1.c": {"safe_v": 2.00, "tax": 0.00},
}

# main.tex Section IV-D: negative in BOTH Summer and Winter -- the group
# the amortization hypothesis was proposed to explain.
FRAGILE_GROUP = {"bs", "cnt", "cubic", "fibcall", "recursion", "select"}


def run_one(bench):
    prof = WORKLOAD_PROFILES[bench]
    safe_v, tax = prof["safe_v"], prof["tax"]

    hybrid_runs = [run_hybrid_once(bench, SCENARIO, s, multiplier=MULTIPLIER, epochs=1) for s in TRAIN_SEEDS]
    avg_hybrid = average_telemetry(hybrid_runs)

    pace_runs = []
    for s in TRAIN_SEEDS:
        agent = train_pace_agent(bench, SCENARIO, s, safe_v, tax, EPOCHS, multiplier=MULTIPLIER)
        pace_runs.append(evaluate_pace_agent(bench, SCENARIO, agent, multiplier=MULTIPLIER, freeze=False, reseed=None))
    avg_pace = average_telemetry(pace_runs)

    if not (avg_hybrid and avg_pace):
        return None

    checkpoint_density = avg_pace.checkpoint_count / avg_hybrid.total_cycles
    t = avg_pace.total_cycles
    waste_fraction = avg_pace.wasted_cycles / t if t > 0 else None
    sleep_fraction = avg_pace.sleep_cycles / t if t > 0 else None  # FIX: recharge loop, previously uncounted
    combined_loss_fraction = ((avg_pace.wasted_cycles + avg_pace.sleep_cycles) / t) if t > 0 else None

    return {
        "benchmark": bench.replace(".c", ""),
        "hybrid_baseline_cycles": avg_hybrid.total_cycles,
        "pace_total_cycles": avg_pace.total_cycles,
        "pace_checkpoint_count": avg_pace.checkpoint_count,
        "pace_wasted_cycles": avg_pace.wasted_cycles,
        "pace_sleep_cycles": avg_pace.sleep_cycles,
        "pace_crash_count": avg_pace.crash_count,
        "checkpoint_density": checkpoint_density,
        "waste_fraction": waste_fraction,
        "sleep_fraction": sleep_fraction,
        "combined_loss_fraction": combined_loss_fraction,
        "fragile_group": bench.replace(".c", "") in FRAGILE_GROUP,
    }


if __name__ == "__main__":
    print(f"{'='*100}\nPHASE 2 ABLATION -- Short-Workload Amortization Hypothesis (Scenario: {SCENARIO})\n{'='*100}\n")

    rows = []
    for bench in WORKLOAD_PROFILES:
        print(f"[+] {bench} ...", flush=True)
        r = run_one(bench)
        if r is None:
            print("    SKIPPED")
            continue
        rows.append(r)
        wf_str = f"{r['waste_fraction']:.4f}" if r["waste_fraction"] is not None else "N/A"
        print(f"    length={r['hybrid_baseline_cycles']:>12,} | ckpt_density={r['checkpoint_density']:.6f} | "
              f"waste_fraction={wf_str} | fragile={r['fragile_group']}")

    fragile_rows = [r for r in rows if r["fragile_group"]]
    other_rows = [r for r in rows if not r["fragile_group"]]

    print(f"\n{'='*100}\nGROUP COMPARISON\n{'='*100}")
    for label, group in [("FRAGILE (bs,cnt,cubic,fibcall,recursion,select)", fragile_rows), ("EVERYTHING ELSE", other_rows)]:
        cd = [r["checkpoint_density"] for r in group]
        wf = [r["waste_fraction"] for r in group if r["waste_fraction"] is not None]
        sf = [r["sleep_fraction"] for r in group if r["sleep_fraction"] is not None]
        clf = [r["combined_loss_fraction"] for r in group if r["combined_loss_fraction"] is not None]
        print(f"\n{label} (n={len(group)}):")
        print(f"  mean checkpoint_density   = {np.mean(cd):.6f}  (median {np.median(cd):.6f})")
        print(f"  mean waste_fraction       = {np.mean(wf):.4f}  (median {np.median(wf):.4f})" if wf else "  waste_fraction: N/A")
        print(f"  mean sleep_fraction       = {np.mean(sf):.4f}  (median {np.median(sf):.4f})" if sf else "  sleep_fraction: N/A")
        print(f"  mean combined_loss_frac   = {np.mean(clf):.4f}  (median {np.median(clf):.4f})" if clf else "  combined_loss_fraction: N/A")
        print(f"  mean length (cycles)      = {np.mean([r['hybrid_baseline_cycles'] for r in group]):,.0f}")

    # ---- Continuous-trend correlation across all 22 (not just the 2-group split) ----
    lengths = np.array([r["hybrid_baseline_cycles"] for r in rows])
    densities = np.array([r["checkpoint_density"] for r in rows])
    wastes = np.array([r["waste_fraction"] for r in rows if r["waste_fraction"] is not None])
    lengths_for_waste = np.array([r["hybrid_baseline_cycles"] for r in rows if r["waste_fraction"] is not None])
    combined = np.array([r["combined_loss_fraction"] for r in rows if r["combined_loss_fraction"] is not None])
    lengths_for_combined = np.array([r["hybrid_baseline_cycles"] for r in rows if r["combined_loss_fraction"] is not None])

    corr_density = np.corrcoef(lengths, densities)[0, 1]
    corr_waste = np.corrcoef(lengths_for_waste, wastes)[0, 1] if len(wastes) > 1 else float("nan")
    corr_combined = np.corrcoef(lengths_for_combined, combined)[0, 1] if len(combined) > 1 else float("nan")

    print(f"\n{'='*100}\nCORRELATION WITH BENCHMARK LENGTH (all {len(rows)} benchmarks)\n{'='*100}")
    print(f"corr(length, checkpoint_density)     = {corr_density:+.3f}  "
          f"(negative = shorter programs need proportionally MORE checkpoints -> supports amortization hypothesis)")
    print(f"corr(length, waste_fraction)         = {corr_waste:+.3f}  "
          f"(crash-recompute cycles only, EXCLUDES recharge/sleep time)")
    print(f"corr(length, combined_loss_fraction) = {corr_combined:+.3f}  "
          f"(wasted+sleep cycles -- the metric that actually matters, since sleep/recharge can dominate total_cycles for short benchmarks; negative = shorter programs lose proportionally more to crash+recharge)")

    with open("phase2_ablation_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n[*] Saved phase2_ablation_results.csv")