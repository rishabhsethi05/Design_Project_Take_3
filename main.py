import os
import pandas as pd
from src.controller.sim_engine import SimulationEngine


def run_experiment(scenario_name="Winter", epochs=30):
    """
    Orchestrates the simulation, handles file paths, and logs results.
    """
    sim = SimulationEngine(scenario=scenario_name)

    # Path Setup for Research Benchmark
    base_path = os.path.dirname(os.path.abspath(__file__))
    benchmark_path = os.path.join(base_path, "data", "benchmarks", "crc.c")

    print(f"\n{'=' * 60}")
    print(f"RUNNING EXPERIMENT: Scenario={scenario_name} | Target=MSP430FR6989")
    print(f"{'=' * 60}")

    # Load Real Benchmark or Fallback
    if os.path.exists(benchmark_path):
        print(f"[*] Loading physical benchmark from: {benchmark_path}")
        code_to_run = sim.parser.load_c_file(benchmark_path)
    else:
        print("[!] crc.c not found. Using Internal CRC-16 Trace.")
        code_to_run = [
                          "uint16_t crc = 0xFFFF;",
                          "crc ^= data[i];",
                          "if (crc & 1) crc = (crc >> 1) ^ 0xA001;",
                          "else crc = crc >> 1;",
                          "data_buffer[i] = crc;",
                          "i++;"
                      ] * 50

    print(f"[*] Instructions in Buffer: {len(code_to_run)}")
    print(f"[*] Starting Training for {epochs} Epochs...")

    # Execute Simulation
    sim.run_simulation(code_to_run, epochs=epochs)

    # Save Results
    output_file = f"output_trace_{scenario_name.lower()}.csv"
    sim.save_trace(output_file)

    # --- Safety Check for Data Analysis ---
    df = pd.read_csv(output_file)

    # Extract data for specific epochs
    first_epoch_data = df[df['Epoch'] == 0]
    final_epoch_data = df[df['Epoch'] == (epochs - 1)]

    # Handle cases where the system never finished (common in 'Night' scenario)
    if first_epoch_data.empty or final_epoch_data.empty:
        print(f"\n[!] RESULT SUMMARY for {scenario_name}:")
        print("    System failed to complete the task (Insufficient Energy/Inflow).")
        return 0, 0, "0.00% (Failed)"
    else:
        first_val = first_epoch_data['Total_Cycles'].max()
        final_val = final_epoch_data['Total_Cycles'].max()

        # Calculate Gain
        if first_val > 0:
            gain_val = ((first_val - final_val) / first_val) * 100
            gain_str = f"{gain_val:.2f}%"
        else:
            gain_val = 0
            gain_str = "0.00%"

        print(f"\nRESULT SUMMARY for {scenario_name}:")
        print(f"Initial Execution Time: {int(first_val)} cycles")
        print(f"Optimized Execution Time: {int(final_val)} cycles")
        print(f"Efficiency Gain: {gain_str}")
        print(f"{'=' * 60}\n")

        return int(first_val), int(final_val), gain_str


if __name__ == "__main__":
    # Robust directory creation for Windows
    for folder in ["data/benchmarks", "logs"]:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError:
                pass

    final_results = []
    scenarios = ["Summer", "Winter", "Night"]

    for sc in scenarios:
        # 1. Run and catch the results
        init_t, opt_t, gain_s = run_experiment(scenario_name=sc, epochs=30)

        # 2. Append for final table
        final_results.append({
            "Scenario": sc,
            "Initial": init_t,
            "Optimized": opt_t,
            "Gain": gain_s
        })

    # Print the Final Research Comparison Table
    print("\n" + "=" * 70)
    print("FINAL RESEARCH COMPARISON: ADAPTIVE ML CHECKPOINTING (MAPI-PRO)")
    print("=" * 70)
    print(f"{'Scenario':<15} | {'Initial (Cycles)':<18} | {'Optimized (Cycles)':<18} | {'Gain'}")
    print("-" * 70)
    for res in final_results:
        print(f"{res['Scenario']:<15} | {res['Initial']:<18} | {res['Optimized']:<18} | {res['Gain']}")
    print("=" * 70)