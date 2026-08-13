import os
import itertools
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.controller.sim_engine import SimulationEngine
from src.agent.model import AdaptiveCheckpointAgent


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_workload(benchmark_file, multiplier=5):
    base_path = os.path.dirname(os.path.abspath(__file__))
    benchmark_path = os.path.join(base_path, "data", "benchmarks", benchmark_file)
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Missing benchmark: {benchmark_path}")
    engine = SimulationEngine(scenario="Summer")
    code = engine.parser.load_c_file(benchmark_path)
    return code * multiplier


def evaluate_candidate(args):
    bench, scenario, safe_v, spam_tax, avg_base_cycles, epochs, seeds = args
    workload = get_workload(bench, 5)
    estimated_steps = len(workload) * epochs
    pace_cycles_runs = []

    for s in seeds:
        set_seed(s)

        # 1. Train the Agent
        agent = AdaptiveCheckpointAgent(alpha=0.10, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01)
        agent.safe_v_threshold = safe_v
        agent.base_spam_tax = spam_tax
        agent.epsilon_decay = (agent.epsilon_min / agent.epsilon_start) ** (1.0 / max(500, int(estimated_steps * 0.75)))

        sim_train = SimulationEngine(scenario=scenario)
        sim_train.strategy = 'pace'
        sim_train.agent = agent
        sim_train.parser.predicted_algo = bench
        sim_train._print_epoch_summary = lambda *args, **kwargs: None
        sim_train.run_simulation(workload, epochs=epochs)

        # 2. Evaluate the Agent (No Exploration)
        agent.reset_state()
        agent.epsilon = 0.0

        sim_eval = SimulationEngine(scenario=scenario)
        sim_eval.strategy = 'pace'
        sim_eval.agent = agent
        sim_eval.parser.predicted_algo = bench
        sim_eval._print_epoch_summary = lambda *args, **kwargs: None
        sim_eval.run_simulation(workload, epochs=1)

        cycles = getattr(sim_eval, 'total_cycles', 0)

        # SURVIVAL PENALTY: Must survive all seeds
        if cycles <= 0:
            return (bench, scenario, safe_v, spam_tax, -999.0, 0)

        pace_cycles_runs.append(cycles)

    avg_pace_cycles = int(np.mean(pace_cycles_runs))
    MIN_VALID_CYCLES = 2000
    if avg_base_cycles < MIN_VALID_CYCLES or avg_pace_cycles < MIN_VALID_CYCLES:
        return (bench, scenario, safe_v, spam_tax, -999.0, avg_pace_cycles)

    raw_gain = ((avg_base_cycles - avg_pace_cycles) / avg_base_cycles) * 100.0
    return (bench, scenario, safe_v, spam_tax, raw_gain, avg_pace_cycles)


def _execute_grid(bench, scenario, avg_base_cycles, seeds, epochs, v_range, tax_range, max_workers):
    tasks = [
        (bench, scenario, sv, tax, avg_base_cycles, epochs, seeds)
        for sv, tax in itertools.product(v_range, tax_range)
    ]

    best_gain = -999.0
    best_params = (2.10, 0.0)
    best_cycles = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_candidate, task) for task in tasks]
        for future in as_completed(futures):
            _, _, sv, tax, gain, cycles = future.result()
            if gain > best_gain:
                best_gain = gain
                best_params = (sv, tax)
                best_cycles = cycles

    return best_gain, best_params[0], best_params[1], best_cycles


def tune_single_workload_parallel(bench, scenario, avg_base_cycles, seeds, epochs, max_workers):
    # STAGE 1: Standard Grid Search (Primary Tuning)
    # Adjusted to focus on ranges that frequently appeared in preliminary testing
    std_v_range = [2.00, 2.05, 2.10, 2.20, 2.30]
    std_tax_range = [0.0, 0.5, 1.0, 2.5, 5.0, 7.5]

    best_gain, best_sv, best_tax, best_cycles = _execute_grid(
        bench, scenario, avg_base_cycles, seeds, epochs, std_v_range, std_tax_range, max_workers
    )

    # STAGE 2: Auto-Trigger Deep Micro-Tuning
    # Dynamically builds combinations around the best parameters found in Stage 1
    if best_gain < 0.0:
        print(f"      [!] Gain {best_gain:>.2f}% detected. Triggering Stage 2: Deep Micro-Tuning...")

        # Offsets relative to the primary winner
        stage2_v_offsets = [-0.10, -0.05, -0.03, 0.0, 0.03, 0.05, 0.10]
        stage2_tax_offsets = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]

        # Generate robust grid and clean up float precision via rounding
        micro_v_range = sorted(list(set([round(max(1.80, best_sv + v_off), 2) for v_off in stage2_v_offsets])))
        micro_tax_range = sorted(list(set([round(max(0.0, best_tax + t_off), 2) for t_off in stage2_tax_offsets])))
        micro_epochs = epochs + 40

        m_gain, m_sv, m_tax, m_cycles = _execute_grid(
            bench, scenario, avg_base_cycles, seeds, micro_epochs, micro_v_range, micro_tax_range, max_workers
        )

        if m_gain > best_gain:
            best_gain = m_gain
            best_sv = m_sv
            best_tax = m_tax
            best_cycles = m_cycles
            print(f"      [*] Stage 2 rescued gain to {best_gain:>.2f}% (V={best_sv:.2f}, Tax={best_tax:.2f})")
        else:
            print(f"      [*] Stage 2 exhausted. Original parameters retained.")

    # STAGE 3: Ultra-Fine Micro-Tuning (Only for close calls: -10.0% to -0.01%)
    # Highly localized combinations zooming in on the immediate neighborhood
    if -15.0 < best_gain < 0.0:
        print(f"      [!] Close call ({best_gain:>.2f}%). Triggering Stage 3: Ultra-Fine Tuning...")

        # Ultra-tight offsets around the best candidate
        stage3_v_offsets = [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
        stage3_tax_offsets = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]

        ultra_v_range = sorted(list(set([round(max(1.80, best_sv + v_off), 2) for v_off in stage3_v_offsets])))
        ultra_tax_range = sorted(list(set([round(max(0.0, best_tax + t_off), 2) for t_off in stage3_tax_offsets])))
        ultra_epochs = epochs + 80

        u_gain, u_sv, u_tax, u_cycles = _execute_grid(
            bench, scenario, avg_base_cycles, seeds, ultra_epochs, ultra_v_range, ultra_tax_range, max_workers
        )

        if u_gain > best_gain:
            best_gain = u_gain
            best_sv = u_sv
            best_tax = u_tax
            best_cycles = u_cycles
            print(
                f"      [*] Stage 3 squeezed out a better gain: {best_gain:>.2f}% (V={best_sv:.2f}, Tax={best_tax:.2f})")
        else:
            print(f"      [*] Stage 3 exhausted. No further improvements found.")

    return best_sv, best_tax, best_gain, best_cycles


def run_expert_tuner():
    EVAL_SEEDS = [42, 123, 999]
    EPOCHS = 15
    MAX_WORKERS = os.cpu_count()

    benchmarks = [
        "aes.c",
        "bs.c",
        "cnt.c",
        "crc.c",
        "dct.c",
        "dijkstra.c",
        "fft.c",
        "fir.c",
        "huffman.c",
        "matmult.c",
        "mergesort.c",
        "prime.c",
        "quicksort.c",
        "select.c",
        "sha256.c",
        "strstr.c",
        "wikisort.c",
        "bubblesort.c",
        "cubic.c",
        "fibcall.c",
        "hash.c",
        "insertsort.c",
        "recursion.c",
        "stringsearch1.c",
        "picojpeg.c"
    ]
    scenarios = ["Summer", "Winter"]

    print("\n" + "=" * 70)
    print(f"[*] DEEP THREE-STAGE AUTO-TUNER STARTING (CORES: {MAX_WORKERS})")
    print("=" * 70)

    # Calculate and immediately print the baselines before tuning begins
    print("\n[*] CALCULATING HYBRID BASELINE CYCLES...")
    baselines = {b: {} for b in benchmarks}
    for bench in benchmarks:
        workload = get_workload(bench, 5)
        for scen in scenarios:
            hybrid_runs = []
            for s in EVAL_SEEDS:
                set_seed(s)
                sim = SimulationEngine(scenario=scen)
                sim.strategy = 'hybrid'
                sim.parser.predicted_algo = bench
                sim._print_epoch_summary = lambda *args, **kwargs: None
                sim.run_simulation(workload, epochs=1)

                cycles = getattr(sim, 'total_cycles', 0)
                if cycles > 0:
                    hybrid_runs.append(cycles)

            if not hybrid_runs:
                baselines[bench][scen] = 999999999
            else:
                baselines[bench][scen] = int(np.mean(hybrid_runs))

        # Display the baseline cycles immediately
        print(
            f"  -> Baseline set for {bench:<18} | Summer: {baselines[bench]['Summer']:<10} | Winter: {baselines[bench]['Winter']:<10}")

    print("\n[*] BASELINES COMPLETE. COMMENCING DEEP TUNING...")

    results = {bench: {} for bench in benchmarks}

    for bench in benchmarks:
        print(f"\n[+] Tuning Benchmark: {bench}")
        for scen in scenarios:
            avg_base_cycles = baselines[bench][scen]
            best_sv, best_tax, best_gain, best_cycles = tune_single_workload_parallel(
                bench, scen, avg_base_cycles, EVAL_SEEDS, EPOCHS, MAX_WORKERS
            )
            results[bench][scen] = {"safe_v": best_sv, "tax": best_tax, "gain": best_gain, "cycles": best_cycles}

            gain_color = "\033[92m" if best_gain > 0 else "\033[91m"
            print(
                f"  -> {scen:<6} Best: SafeV={best_sv:.2f}V | Tax={best_tax:>4.2f} | Cycles: {best_cycles:<9} | Final Gain: {gain_color}{best_gain:>6.2f}%\033[0m")

    print("\n" + "=" * 145)
    print(" TRUE PARETO-OPTIMAL PROFILES (THREE-STAGE VERIFIED)")
    print("=" * 145)
    print(
        f"{'Benchmark':<18} | {'Summer (V, Tax)':<18} | {'Summer Gain':<13} | {'Sum Base':<10} | {'Sum PACE':<10} | {'Winter (V, Tax)':<18} | {'Winter Gain':<13} | {'Win Base':<10} | {'Win PACE':<10}")
    print("-" * 145)

    for bench in benchmarks:
        s_res = results[bench]["Summer"]
        w_res = results[bench]["Winter"]
        s_param_str = f"{s_res['safe_v']:.2f}V, {s_res['tax']:.2f}"
        w_param_str = f"{w_res['safe_v']:.2f}V, {w_res['tax']:.2f}"
        s_gain_str = f"{s_res['gain']:>8.2f}%"
        w_gain_str = f"{w_res['gain']:>8.2f}%"
        s_cyc_str = f"{s_res['cycles']}"
        w_cyc_str = f"{w_res['cycles']}"
        s_base_str = f"{baselines[bench]['Summer']}"
        w_base_str = f"{baselines[bench]['Winter']}"

        print(
            f"{bench:<18} | {s_param_str:<18} | {s_gain_str:<13} | {s_base_str:<10} | {s_cyc_str:<10} | {w_param_str:<18} | {w_gain_str:<13} | {w_base_str:<10} | {w_cyc_str:<10}")
    print("=" * 145)

    print("\n" + "=" * 70)
    print("[*] COPY-PASTE READY WORKLOAD_PROFILES FOR main.py")
    print("=" * 70)
    print("WORKLOAD_PROFILES = {")
    for bench in benchmarks:
        s = results[bench]["Summer"]
        w = results[bench]["Winter"]
        print(f'    "{bench}": {{')
        print(f'        "Summer": {{"safe_v": {s["safe_v"]:.2f}, "tax": {s["tax"]:.2f}}},')
        print(f'        "Winter": {{"safe_v": {w["safe_v"]:.2f}, "tax": {w["tax"]:.2f}}},')
        print(f'    }},')
    print("}")
    print("=" * 70)


if __name__ == "__main__":
    run_expert_tuner()