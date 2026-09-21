"""
diagnose_zero_anomaly.py -- investigates the exact 0.00% gain on fir.c
and insertsort.c in Experiment 1 Setup A's frozen-transfer condition.

Hypothesis: both the hybrid baseline and the frozen-Summer-transfer PACE
policy independently hit the recharge loop's 10,000-attempt cap under
Winter for these two benchmarks, and total_cycles collapses to nearly
the same sentinel value (~160,016,000, dominated by 10,001 x 16,000)
regardless of where each one actually died -- producing a fake-looking
"tie" that's really "both failed catastrophically," not equal performance.

This prints PER-SEED (not averaged) raw telemetry, including the new
target_died flag, for both conditions on both suspect benchmarks.
"""

import sys
sys.path.insert(0, ".")

from src.experiment_harness import run_hybrid_once, train_pace_agent, evaluate_pace_agent

SEEDS = [42, 123, 999]
EPOCHS = 15
MULTIPLIER = 5

CASES = [
    {"bench": "fir.c",        "safe_v_summer": 2.00, "tax_summer": 7.50},
    {"bench": "insertsort.c", "safe_v_summer": 1.90, "tax_summer": 0.00},
]

for case in CASES:
    bench = case["bench"]
    print(f"\n{'='*80}\n{bench}\n{'='*80}")

    for s in SEEDS:
        hybrid = run_hybrid_once(bench, "Winter", s, multiplier=MULTIPLIER, epochs=1)
        print(f"\nSeed {s}:")
        print(f"  HYBRID   -> cycles={hybrid.total_cycles:,} | crashes={hybrid.crash_count} | "
              f"target_died={hybrid.target_died} | valid={hybrid.valid}")

        summer_agent = train_pace_agent(bench, "Summer", s, case["safe_v_summer"], case["tax_summer"],
                                         EPOCHS, multiplier=MULTIPLIER)
        transfer = evaluate_pace_agent(bench, "Winter", summer_agent, multiplier=MULTIPLIER,
                                        freeze=True, reseed=s)
        print(f"  TRANSFER -> cycles={transfer.total_cycles:,} | crashes={transfer.crash_count} | "
              f"checkpoints={transfer.checkpoint_count} | target_died={transfer.target_died} | "
              f"valid={transfer.valid}")

        if hybrid.total_cycles == transfer.total_cycles:
            print(f"  *** EXACT MATCH on raw cycles for this seed ***")