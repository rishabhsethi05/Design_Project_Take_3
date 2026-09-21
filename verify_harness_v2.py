"""
verify_harness_v2.py -- confirms the freeze=True hypothesis by running
crc.c both ways and comparing against the known Table IV value.
"""

import sys
sys.path.insert(0, ".")

from src.experiment_harness import (
    run_hybrid_once, train_pace_agent, evaluate_pace_agent,
    average_telemetry, compute_gain
)

SEEDS = [42, 123, 999]
EPOCHS = 15
BENCH, SCENARIO, SAFE_V, TAX = "crc.c", "Summer", 2.00, 1.00
EXPECTED_GAIN = 46.10  # from the original tuner.py log / Table IV

hybrid_runs = [run_hybrid_once(BENCH, SCENARIO, s, epochs=1) for s in SEEDS]
avg_hybrid = average_telemetry(hybrid_runs)
print(f"Hybrid avg cycles: {avg_hybrid.total_cycles:,}")

for freeze in (False, True):
    pace_runs = []
    for s in SEEDS:
        agent = train_pace_agent(BENCH, SCENARIO, s, SAFE_V, TAX, EPOCHS)
        telem = evaluate_pace_agent(BENCH, SCENARIO, agent, freeze=freeze, reseed=None)
        pace_runs.append(telem)
    avg_pace = average_telemetry(pace_runs)
    gain = compute_gain(avg_hybrid.total_cycles, avg_pace.total_cycles) if avg_pace else None
    print(f"\nfreeze={freeze}: PACE avg cycles = {avg_pace.total_cycles:,} | Gain = {gain:.2f}% (expected ~{EXPECTED_GAIN}%)")