"""
main.py -- per-benchmark figure/CSV generation for the PACE paper.

CHANGES MADE IN THIS PASS (read before running):
1. WORKLOAD_PROFILES replaced with the actual final, post-harvester-fix
   tuned parameters (matching the last verified tuner.py run / Table IV
   in main.tex exactly). The previous dict here was stale -- e.g. aes.c
   had tax=0.00/0.00 here vs the real tuned 0.50/1.00 -- and would have
   produced figures inconsistent with the paper's own tables.
2. fft.c, matmult.c, picojpeg.c removed from WORKLOAD_PROFILES. These are
   excluded from the paper's evaluation (checkpoint state size vs.
   capacitor energy budget -- see Section IV-A of main.tex). Do not add
   them back without also updating the paper.
3. multiplier=5 kept as-is -- confirmed to match tuner.py's own
   get_workload(bench, 5), so this is now a documented, intentional part
   of the methodology (main.tex Section IV-A has been updated to state
   this explicitly), not an inconsistency.
4. The Night scenario previously short-circuited to a hardcoded, fake
   "0.00% (Insufficient Power)" result WITHOUT running any simulation.
   main.tex claims Night is used to verify graceful failure under zero
   harvesting -- that claim was not actually true of this code. Night now
   runs through the same pipeline as Summer/Winter. This is safe: the
   recharge loop in sim_engine.py already caps at 10,000 attempts before
   breaking out, so a zero-inflow run terminates rather than hanging.
5. Removed the "(BALANCED PARETO-OPTIMAL PROFILES (TUNED))" comment label
   -- main.tex no longer uses "Pareto-optimal" terminology (the tuning
   pipeline is a scalar best-mean-gain grid search, not multi-objective
   Pareto optimization), so this script shouldn't either.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.controller.sim_engine import SimulationEngine
from src.agent.model import AdaptiveCheckpointAgent

# ============================================================
# ANSI COLOR TERMINAL FORMATTING
# ============================================================
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_fresh_workload(benchmark_file, scenario, multiplier=5):
    """Generates a pristine, unmodified workload to prevent state leakage between runs."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    benchmark_path = os.path.join(base_path, "data", "benchmarks", benchmark_file)

    engine = SimulationEngine(scenario=scenario)
    code = engine.parser.load_c_file(benchmark_path)
    return code * multiplier


# ============================================================
# TUNED SAFETY-THRESHOLD AND PENALTY CONFIGURATIONS
# (post-harvester-fix, matches main.tex Table IV; 22 benchmarks --
#  fft.c/matmult.c/picojpeg.c intentionally excluded, see docstring)
# ============================================================
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


# ============================================================
# MAIN EXPERIMENT FUNCTION (SYNCED WITH TUNER.PY)
# ============================================================
def run_experiment(scenario_name="Winter", epochs=15, benchmark_file="quicksort.c", multiplier=5, seeds=[42, 123, 999]):
    print(f"\n{'=' * 65}")
    print(f"RUNNING EXPERIMENT ({len(seeds)}-SEED AVG): Scenario={scenario_name:<6} | Target={benchmark_file}")
    print(f"{'=' * 65}")

    base_path = os.path.dirname(os.path.abspath(__file__))
    benchmark_path = os.path.join(base_path, "data", "benchmarks", benchmark_file)

    if not os.path.exists(benchmark_path):
        print(f"{COLOR_RED}[!] Warning: File {benchmark_path} not found. Skipping.{COLOR_RESET}")
        return None

    # Load once just to extract the predicted algorithm name for logging
    temp_engine = SimulationEngine(scenario=scenario_name)
    temp_engine.parser.load_c_file(benchmark_path)
    predicted_algo = temp_engine.parser.predicted_algo

    # NOTE: Night is no longer short-circuited to a fake result -- it now
    # runs through the exact same pipeline as Summer/Winter below, so it
    # actually exercises (and can verify) graceful degradation under zero
    # harvesting, matching what main.tex claims about this scenario.
    bench_profile = WORKLOAD_PROFILES.get(benchmark_file, {})
    profile = bench_profile.get(scenario_name, {"safe_v": 2.00, "tax": 0.5})

    hybrid_cycles_runs = []
    pace_cycles_runs = []

    for s in seeds:
        set_seed(s)

        # 1. EVALUATE HYBRID BASELINE (PRISTINE WORKLOAD)
        wl_hybrid = get_fresh_workload(benchmark_file, scenario_name, multiplier)

        sim_hybrid = SimulationEngine(scenario=scenario_name)
        sim_hybrid.strategy = 'hybrid'
        sim_hybrid.parser.predicted_algo = benchmark_file
        sim_hybrid._print_epoch_summary = lambda *args, **kwargs: None
        sim_hybrid.run_simulation(wl_hybrid, epochs=1)

        hybrid_cycles = getattr(sim_hybrid, 'total_cycles', 0)
        if hybrid_cycles > 0:
            hybrid_cycles_runs.append(hybrid_cycles)

        # 2. TRAIN PACE AGENT (PRISTINE WORKLOAD)
        wl_train = get_fresh_workload(benchmark_file, scenario_name, multiplier)

        agent = AdaptiveCheckpointAgent(alpha=0.10, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01)
        agent.safe_v_threshold = profile["safe_v"]
        agent.base_spam_tax = profile["tax"]

        estimated_steps = len(wl_train) * epochs
        agent.epsilon_decay = (agent.epsilon_min / agent.epsilon_start) ** (1.0 / max(500, int(estimated_steps * 0.75)))

        sim_train = SimulationEngine(scenario=scenario_name)
        sim_train.strategy = 'pace'
        sim_train.agent = agent
        sim_train.parser.predicted_algo = benchmark_file
        sim_train._print_epoch_summary = lambda *args, **kwargs: None
        sim_train.run_simulation(wl_train, epochs=epochs)

        # 3. EVALUATE PACE AGENT (PRISTINE WORKLOAD)
        wl_eval = get_fresh_workload(benchmark_file, scenario_name, multiplier)

        agent.reset_state()
        agent.epsilon = 0.0

        sim_eval = SimulationEngine(scenario=scenario_name)
        sim_eval.strategy = 'pace'
        sim_eval.agent = agent
        sim_eval.parser.predicted_algo = benchmark_file
        sim_eval._print_epoch_summary = lambda *args, **kwargs: None
        sim_eval.run_simulation(wl_eval, epochs=1)

        pace_cycles = getattr(sim_eval, 'total_cycles', 0)
        if pace_cycles > 0:
            pace_cycles_runs.append(pace_cycles)

    # 4. AVERAGING & METRICS MATH (Synced MIN_VALID_CYCLES with tuner.py)
    MIN_VALID_CYCLES = 2000

    avg_hybrid_cycles = int(np.mean(hybrid_cycles_runs)) if hybrid_cycles_runs else 0
    avg_pace_cycles = int(np.mean(pace_cycles_runs)) if pace_cycles_runs else 0

    hybrid_crashed = avg_hybrid_cycles < MIN_VALID_CYCLES
    pace_crashed = avg_pace_cycles < MIN_VALID_CYCLES

    if scenario_name == "Night":
        # Both should crash under zero inflow -- this is the graceful-failure
        # check main.tex describes. Report it plainly rather than faking a gain.
        if hybrid_crashed and pace_crashed:
            gain_str = "Both failed gracefully (0 inflow, as expected)"
        elif hybrid_crashed and not pace_crashed:
            gain_str = "UNEXPECTED: PACE survived zero inflow, hybrid did not"
        elif pace_crashed and not hybrid_crashed:
            gain_str = "UNEXPECTED: Hybrid survived zero inflow, PACE did not"
        else:
            gain_str = "UNEXPECTED: both policies completed under zero inflow"
        gain_val = 0.0
        print(f"[Night check] Hybrid: {avg_hybrid_cycles:,} | PACE: {avg_pace_cycles:,} | {gain_str}")
        return {
            "scenario": scenario_name, "hybrid": avg_hybrid_cycles, "pace": avg_pace_cycles,
            "gain_str": gain_str, "gain_numeric": gain_val, "algorithm": predicted_algo,
            "safe_v": profile["safe_v"], "tax": profile["tax"]
        }

    if hybrid_crashed and not pace_crashed:
        gain_str = "Baseline Failed (PACE Won)"
        gain_val = 100.0
    elif pace_crashed and not hybrid_crashed:
        gain_str = "PACE Failed (Baseline Won)"
        gain_val = -100.0
    elif hybrid_crashed and pace_crashed:
        gain_str = "Both Failed (Too Harsh)"
        gain_val = 0.0
    else:
        gain_val = ((avg_hybrid_cycles - avg_pace_cycles) / avg_hybrid_cycles) * 100.0
        gain_str = f"{gain_val:.2f}%"

    # Terminal Log with Color
    if gain_val > 0:
        colored_result = f"{COLOR_GREEN}{gain_str}{COLOR_RESET}"
    elif gain_val < 0:
        colored_result = f"{COLOR_RED}{gain_str}{COLOR_RESET}"
    else:
        colored_result = f"{COLOR_YELLOW}{gain_str}{COLOR_RESET}"

    print(
        f"[{len(seeds)}-Seed Avg] Hybrid: {avg_hybrid_cycles:,} | PACE: {avg_pace_cycles:,} | Result: {colored_result}")

    return {
        "scenario": scenario_name, "hybrid": avg_hybrid_cycles, "pace": avg_pace_cycles,
        "gain_str": gain_str, "gain_numeric": gain_val, "algorithm": predicted_algo,
        "safe_v": profile["safe_v"], "tax": profile["tax"]
    }


# ============================================================
# VISUALIZATION & EXPORT
# ============================================================
def generate_bar_chart(results, algorithm_name):
    # Night excluded from this chart -- it isn't a cycle-count comparison,
    # see the Night-specific handling in run_experiment above.
    valid_results = [r for r in results if r['scenario'] != 'Night']
    if not valid_results:
        return

    scenarios = [r['scenario'] for r in valid_results]
    hybrid_cycles = [r['hybrid'] for r in valid_results]
    pace_cycles = [r['pace'] for r in valid_results]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, hybrid_cycles, width, label='Hybrid Baseline', color='#e07a5f')
    ax.bar(x + width / 2, pace_cycles, width, label='PACE (Proposed)', color='#3d5a80')

    ax.set_ylabel('Mean Execution Cycles', fontsize=12)
    ax.set_title(f'Performance Comparison: PACE vs Hybrid Baseline\nBenchmark: {algorithm_name}', fontsize=14,
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)

    max_val = max(max(hybrid_cycles), max(pace_cycles)) if (hybrid_cycles and pace_cycles) else 10000

    for i, v in enumerate(hybrid_cycles):
        ax.text(i - width / 2, v + (max_val * 0.01), f"{v:,}", ha='center', va='bottom', fontsize=9, rotation=35)
    for i, v in enumerate(pace_cycles):
        ax.text(i + width / 2, v + (max_val * 0.01), f"{v:,}", ha='center', va='bottom', fontsize=9, rotation=35)

    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0, max_val * 1.25)
    fig.tight_layout()

    filename = f"bar_chart_comparison_{algorithm_name.replace('.c', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[*] Saved Bar Chart: {filename}")


if __name__ == "__main__":
    WORKLOAD_MULTIPLIER = 5  # Confirmed to match tuner.py -- see module docstring point 3.
    EVAL_SEEDS = [42, 123, 999]  # Synced to tuner.py
    EPOCHS = 15  # Reverted back to tuner.py default to fix decay mismatch
    scenarios = ["Summer", "Winter", "Night"]

    all_global_results = []

    print(f"\n{'#' * 85}")
    print(f"[*] STARTING FULL SUITE EVALUATION ({len(WORKLOAD_PROFILES)} BENCHMARKS | {len(EVAL_SEEDS)}-SEED AVERAGE)")
    print(f"{'#' * 85}\n")

    for benchmark in WORKLOAD_PROFILES.keys():
        benchmark_results = []
        for sc in scenarios:
            result = run_experiment(
                scenario_name=sc,
                epochs=EPOCHS,
                benchmark_file=benchmark,
                multiplier=WORKLOAD_MULTIPLIER,
                seeds=EVAL_SEEDS
            )
            if result:
                benchmark_results.append(result)
                all_global_results.append((benchmark, result))

        generate_bar_chart(benchmark_results, benchmark)

    print("\n[*] Generating Complete CSV Data Sheet...")
    csv_rows = []

    for bench_name, res in all_global_results:
        clean_bench_name = bench_name.replace(".c", "")
        csv_rows.append({
            "Benchmark": clean_bench_name,
            "Algorithm_Class": res["algorithm"],
            "Scenario": res["scenario"],
            "Hybrid_Avg_Cycles": res["hybrid"],
            "PACE_Avg_Cycles": res["pace"],
            "Applied_Safe_V": res["safe_v"],
            "Applied_Tax": res["tax"],
            "PACE_Gain": res["gain_str"]
        })

    df = pd.DataFrame(csv_rows)
    csv_filename = "true_simulation_results_ALL.csv"
    df.to_csv(csv_filename, index=False)
    print(f"[*] Saved Full Evaluation CSV: {csv_filename}")

    print("\n[*] Generating Copy-Paste Spreadsheet Data...")
    spreadsheet_dict = {}

    for bench_name, res in all_global_results:
        clean_bench_name = bench_name.replace(".c", "")
        if res["scenario"] == "Night":
            continue  # Night isn't a cycle-comparison row -- see CSV above for its result instead.
        if clean_bench_name not in spreadsheet_dict:
            spreadsheet_dict[clean_bench_name] = {
                "Benchmarks": clean_bench_name,
                "Hybrid Summer": 0, "Hybrid Winter": 0,
                "Proposed Sumr": 0, "Proposed Winter": 0
            }

        if res["scenario"] == "Summer":
            spreadsheet_dict[clean_bench_name]["Hybrid Summer"] = res["hybrid"]
            spreadsheet_dict[clean_bench_name]["Proposed Sumr"] = res["pace"]
        elif res["scenario"] == "Winter":
            spreadsheet_dict[clean_bench_name]["Hybrid Winter"] = res["hybrid"]
            spreadsheet_dict[clean_bench_name]["Proposed Winter"] = res["pace"]

    df_spreadsheet = pd.DataFrame(list(spreadsheet_dict.values()))
    sheet_filename = "spreadsheet_raw_cycles.csv"
    df_spreadsheet.to_csv(sheet_filename, index=False)
    print(f"[*] Saved Copy-Paste Spreadsheet CSV: {sheet_filename}")

    # ============================================================
    # FINAL MASTER CONSOLE REPORT WITH ANSI COLOR HIGHLIGHTS
    # ============================================================
    print("\n" + "═" * 115)
    print(
        f" FINAL RESEARCH SUMMARY: PACE CHECKPOINTING FRAMEWORK (ALL {len(WORKLOAD_PROFILES)} BENCHMARKS | {len(EVAL_SEEDS)}-SEED AVG)")
    print(f" Workload Expansion: {WORKLOAD_MULTIPLIER}x | Evaluated Seeds: {EVAL_SEEDS}")
    print("═" * 115)
    print(
        f"{'Benchmark':<18}| {'Scenario':<10}| {'Hybrid Avg Cycles':<18}| {'PACE Avg Cycles':<18}| {'SafeV':<7}| {'Tax':<6}| {'Gain'}")
    print("─" * 115)

    for bench_name, res in all_global_results:
        safe_v_str = f"{res['safe_v']:.2f}V" if isinstance(res['safe_v'], (int, float)) else str(res['safe_v'])
        tax_str = f"{res['tax']:.1f}" if isinstance(res['tax'], (int, float)) else str(res['tax'])

        padded_gain_str = f"{res['gain_str']:<40}"

        if res['gain_numeric'] > 0:
            colored_gain = f"{COLOR_GREEN}{padded_gain_str}{COLOR_RESET}"
        elif res['gain_numeric'] < 0:
            colored_gain = f"{COLOR_RED}{padded_gain_str}{COLOR_RESET}"
        else:
            colored_gain = f"{COLOR_YELLOW}{padded_gain_str}{COLOR_RESET}"

        print(
            f"{bench_name:<18}| {res['scenario']:<10}| {res['hybrid']:<18,}| {res['pace']:<18,}| {safe_v_str:<7}| {tax_str:<6}| {colored_gain}")

        if res['scenario'] == "Night":
            print("-" * 115)

    print("═" * 115)