import os
import pandas as pd
import random
import numpy as np
from src.controller.sim_engine import SimulationEngine

# Setting seeds for consistent, reproducible research results
random.seed(42)
np.random.seed(42)


def run_experiment(scenario_name="Winter", epochs=100, benchmark_file="quicksort.c", multiplier=1):
    """
    Orchestrates the training and evaluation of the MAPI-PRO agent.
    """
    sim = SimulationEngine(scenario=scenario_name)

    # Path Setup
    base_path = os.path.dirname(os.path.abspath(__file__))
    benchmark_path = os.path.join(base_path, "data", "benchmarks", benchmark_file)

    print(f"\n{'=' * 60}")
    print(f"RUNNING EXPERIMENT: Scenario={scenario_name} | Target=MSP430FR6989")
    print(f"{'=' * 60}")

    if os.path.exists(benchmark_path):
        print(f"[*] Loading benchmark: {benchmark_file}")
        single_pass = sim.parser.load_c_file(benchmark_path)
        code_to_run = single_pass * multiplier
        print(f"[*] Identified Algorithm: {sim.parser.predicted_algo}")
    else:
        print(f"[!] Benchmark not found. Check path: {benchmark_path}")
        return 0, 0, "N/A"

    print(f"[*] Workload: {len(code_to_run)} instructions ({multiplier}x expansion)")
    print(f"[*] Training Agent for {epochs} Epochs...")

    # EXECUTE SIMULATION
    # The engine now trains for N epochs and calculates Baseline internally
    sim.run_simulation(code_to_run, epochs=epochs)

    # RECOVERY & ANALYSIS
    try:
        # --- REVERTED: Pulling reliable data from the updated Engine ---
        initial_cycles = sim.baseline_cycles
        optimized_cycles = sim.optimized_cycles

        # Ensure we don't show negative gains or divide by zero for Night
        if scenario_name == "Night" or optimized_cycles <= 0 or initial_cycles <= 0:
            gain_str = "0.00% (Insufficient Power)"
            initial_cycles = 0
            optimized_cycles = 0
        else:
            # Calculation: (Baseline - Optimized) / Baseline
            gain_val = ((initial_cycles - optimized_cycles) / initial_cycles) * 100
            gain_str = f"{gain_val:.2f}%"

        print(f"\nRESULT SUMMARY for {scenario_name}:")
        print(f"Initial (Baseline): {int(initial_cycles)} cycles")
        print(f"Optimized (ML):      {int(optimized_cycles)} cycles")
        print(f"Efficiency Gain:     {gain_str}")
        print(f"{'=' * 60}\n")

        return int(initial_cycles), int(optimized_cycles), gain_str

    except Exception as e:
        print(f"[!] Error during analysis: {e}")
        return 0, 0, "Error"


if __name__ == "__main__":
    # --- CONFIGURATION: Match your benchmark and workload expansion ---
    SELECTED_BENCHMARK = ("dijkstra.c")
    WORKLOAD_MULTIPLIER = 5

    scenarios = ["Summer", "Winter", "Night"]
    final_results = []

    for sc in scenarios:
        init_t, opt_t, gain_s = run_experiment(
            scenario_name=sc,
            epochs=100,
            benchmark_file=SELECTED_BENCHMARK,
            multiplier=WORKLOAD_MULTIPLIER
        )
        final_results.append({
            "Scenario": sc,
            "Initial": init_t,
            "Optimized": opt_t,
            "Gain": gain_s
        })

    # FINAL TABULAR REPORT
    print("\n" + "═" * 75)
    print("  FINAL RESEARCH SUMMARY: ADAPTIVE ML CHECKPOINTING (MAPI-PRO)")
    print(f"  Target: {SELECTED_BENCHMARK.upper()} | Workload: {WORKLOAD_MULTIPLIER}x")
    print("═" * 75)
    print(f"  {'Scenario':<15} | {'Baseline (Cycles)':<18} | {'MAPI-PRO (Cycles)':<18} | {'Gain'}")
    print("─" * 75)
    for res in final_results:
        print(f"  {res['Scenario']:<15} | {res['Initial']:<18} | {res['Optimized']:<18} | {res['Gain']}")
    print("═" * 75)