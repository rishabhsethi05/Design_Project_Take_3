"""
verify_harness.py -- sanity check for experiment_harness.py.

Reproduces two known-good Table IV values (one benchmark that was
already stable across our two conflicting runs, one that was a sign-flip
case) to confirm the harness's seeding fix actually works before we wire
it into tuner.py for real.

Expected (from the original, trusted tuner.py log -- Table IV in main.tex):
  aes.c  Summer  safe_v=2.00 tax=0.50  epochs=15  seeds=[42,123,999] -> ~97.09% gain
  crc.c  Summer  safe_v=2.00 tax=1.00  epochs=15  seeds=[42,123,999] -> ~46.10% gain
  (crc is the one that flipped to -10.77% in the buggy main_figures.py run --
   if the harness is fixed, this should land back near +46%, not near -11%.)

Run this from your project root (same place you run tuner.py from).
"""

import sys
sys.path.insert(0, ".")  # adjust if your project root isn't the CWD

from src.experiment_harness import run_hybrid_once, run_rung_once, average_telemetry, compute_gain

SEEDS = [42, 123, 999]
EPOCHS = 15

TEST_CASES = [
    {"bench": "aes.c", "scenario": "Summer", "safe_v": 2.00, "tax": 0.50, "expected_gain": 97.09},
    {"bench": "crc.c", "scenario": "Summer", "safe_v": 2.00, "tax": 1.00, "expected_gain": 46.10},
]

for case in TEST_CASES:
    bench, scenario, safe_v, tax = case["bench"], case["scenario"], case["safe_v"], case["tax"]
    print(f"\n{'='*70}\n{bench} / {scenario}  (safe_v={safe_v}, tax={tax})\n{'='*70}")

    # Hybrid baseline -- each seed independently, exactly like tuner.py
    hybrid_runs = [run_hybrid_once(bench, scenario, s, epochs=1) for s in SEEDS]
    avg_hybrid = average_telemetry(hybrid_runs)
    print(f"Hybrid avg cycles: {avg_hybrid.total_cycles:,}" if avg_hybrid else "Hybrid: FAILED")

    # PACE, full rung (R4) -- train+eval per seed via the harness
    pace_runs = [
        run_rung_once(bench, scenario, s, safe_v, tax, EPOCHS, use_reflexes=True, use_agent=True)
        for s in SEEDS
    ]
    avg_pace = average_telemetry(pace_runs)
    print(f"PACE avg cycles:   {avg_pace.total_cycles:,}" if avg_pace else "PACE: FAILED")

    if avg_hybrid and avg_pace:
        gain = compute_gain(avg_hybrid.total_cycles, avg_pace.total_cycles)
        print(f"\nGain: {gain:.2f}%   (expected ~{case['expected_gain']}%)")
        diff = abs(gain - case["expected_gain"])
        if diff < 2.0:
            print("PASS -- matches Table IV within tolerance.")
        else:
            print(f"MISMATCH -- off by {diff:.2f} points. Harness may still have an issue, investigate before trusting it.")
    else:
        print("Could not compute gain -- one side failed to produce a valid run.")