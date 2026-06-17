import os
import sys
import random
import traceback
from src.controller.sim_engine import SimulationEngine


# ============================================================
# BULLETPROOF SILENT WRAPPER (Forces UTF-8)
# ============================================================
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        # Force UTF-8 encoding to prevent Windows charmap crashes
        self._devnull = open(os.devnull, 'w', encoding='utf-8')
        sys.stdout = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self._devnull.close()


# ============================================================
# HEADLESS SIMULATION EVALUATOR
# ============================================================
def evaluate_pace(reward_config, benchmark_file="quicksort.c", multiplier=1, epochs=400):
    base_path = os.path.dirname(os.path.abspath(__file__))
    benchmark_path = os.path.join(base_path, "data", "benchmarks", benchmark_file)

    total_combined_cycles = 0

    for scenario in ["Summer", "Winter"]:
        sim_pace = SimulationEngine(scenario=scenario, reward_config=reward_config)
        setattr(sim_pace, 'strategy', 'pace')

        err_msg = None
        trace = None

        with HiddenPrints():
            single_pass = sim_pace.parser.load_c_file(benchmark_path)
            code_to_run = single_pass * multiplier

            try:
                sim_pace.run_simulation(code_to_run, epochs=epochs)
                cycles = getattr(sim_pace, 'total_cycles', getattr(sim_pace, 'optimized_cycles', 0))
            except Exception as e:
                err_msg = str(e)
                trace = traceback.format_exc()
                cycles = 0

                # Print errors OUTSIDE the HiddenPrints block to protect the I/O stream
        if err_msg and "Blackout" not in err_msg:
            print(f"\n[!] FATAL PYTHON ERROR DETECTED: {err_msg}")
            print(trace)

        if cycles == 0:
            return float('inf')

        total_combined_cycles += cycles

    return total_combined_cycles


# ============================================================
# DIAGNOSTIC RANDOM SEARCH OPTIMIZER
# ============================================================
def tune_rewards(iterations=250):
    print(f"\n{'=' * 60}")
    print("🚀 STARTING DIAGNOSTIC REWARD OPTIMIZER")
    print(f"[*] Target: quicksort.c | Expansion: 1x | Epochs: 400")
    print(f"{'=' * 60}\n")

    best_score = float('inf')
    best_rewards = None

    for i in range(iterations):
        if i == 0:
            test_rewards = {
                "crash": -10000.0,
                "summer_cp": -500.0,
                "winter_cp": -50.0,
                "execute": 10.0
            }
            print("[*] Iteration 1: Testing Baseline Sanity Check...")
        else:
            test_rewards = {
                "crash": random.uniform(-15000.0, -5000.0),
                "summer_cp": random.uniform(-1000.0, -100.0),
                "winter_cp": random.uniform(-200.0, -10.0),
                "execute": random.uniform(-5.0, 15.0)
            }

        total_cycles = evaluate_pace(reward_config=test_rewards)

        if total_cycles < best_score:
            best_score = total_cycles
            best_rewards = test_rewards
            print(f"🏆 NEW OPTIMAL FOUND AT ITERATION {i + 1}!")
            print(f" -> Summer + Winter Cycles: {int(best_score)}")
            print(
                f" -> Values: {{'crash': {best_rewards['crash']:.1f}, 'summer_cp': {best_rewards['summer_cp']:.1f}, 'winter_cp': {best_rewards['winter_cp']:.1f}, 'execute': {best_rewards['execute']:.1f}}}\n")

        if (i + 1) % 10 == 0:
            print(f"[*] Completed {i + 1}/{iterations} iterations...")

    print(f"\n{'=' * 60}")
    print("🎯 OPTIMIZATION COMPLETE")
    if best_score == float('inf'):
        print("[!] NO OPTIMAL SOLUTION FOUND. The agent crashed on every attempt.")
    else:
        print(f"Lowest Combined Cycles: {int(best_score)}")
        print("Copy this block into model.py:")
        print(
            f"self.reward_config = {{\n    'crash': {best_rewards['crash']:.1f},\n    'summer_cp': {best_rewards['summer_cp']:.1f},\n    'winter_cp': {best_rewards['winter_cp']:.1f},\n    'execute': {best_rewards['execute']:.1f}\n}}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    tune_rewards(iterations=100)