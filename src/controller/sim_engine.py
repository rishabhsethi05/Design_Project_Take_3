import pandas as pd
import numpy as np
from src.environment.capacitor import MSP430Capacitor
from src.environment.harvester import EnergyHarvester
from src.profiler.parser import MapiProParser
from src.agent.model import AdaptiveCheckpointAgent

class SimulationEngine:
    def __init__(self, scenario="Winter"):
        # Initialize hardware and agent modules
        self.cap = MSP430Capacitor()
        self.harvester = EnergyHarvester(scenario=scenario)
        self.parser = MapiProParser()
        self.agent = AdaptiveCheckpointAgent()

        # Stats tracking
        self.trace_logs = []
        self.total_cycles = 0
        self.last_checkpoint_pc = 0
        self.recomputations = 0

    def run_simulation(self, code_trace, epochs=1):
        """
        Runs the hardware-aware simulation.
        code_trace: List of dicts containing {'code': str, 'line_no': int}
        """
        for epoch in range(epochs):
            self.cap.reset_to_full()
            pc = 0
            self.total_cycles = 0
            self.last_checkpoint_pc = 0
            # Buffer to store checkpoint details for the final "Take 2 Style" summary
            epoch_checkpoints = []

            while pc < len(code_trace):
                # 1. Fetch current instruction data
                instr_data = code_trace[pc]
                line_code = instr_data["code"]
                original_line_no = instr_data["line_no"]

                v_before = self.cap.get_current_voltage()
                inflow = self.harvester.get_inflow_power()
                pc_percent = (pc / len(code_trace)) * 100

                # 2. Agent decides: 0 = Execute, 1 = Checkpoint
                action = self.agent.choose_action(v_before, pc_percent, inflow)

                energy_spent = 0
                cycle_cost = 0
                status = "SUCCESS"

                if action == 1:  # ACTION: CHECKPOINT
                    energy_spent, cycle_cost = self.parser.get_checkpoint_cost(32)
                    self.last_checkpoint_pc = pc
                    # Log CP with its real line number and code content for summary
                    epoch_checkpoints.append({
                        "line": original_line_no,
                        "content": line_code
                    })
                else:  # ACTION: EXECUTE
                    energy_spent, cycle_cost = self.parser.profile_line(line_code)
                    pc += 1

                # 3. Apply Physics (Capacitor discharge + Harvester inflow)
                self.cap.consume_energy(energy_spent)
                # Step harvest based on cycle duration (Normalization)
                self.harvester.step_harvest(self.cap, (cycle_cost / 16000))
                self.total_cycles += cycle_cost
                v_after = self.cap.get_current_voltage()

                # 4. Handle Physical Failure (Brown-out/Blackout)
                if self.cap.is_dead:
                    status = "CRASH"
                    pc = self.last_checkpoint_pc
                    self.recomputations += 1

                    # Simulated recharge logic (MSP430 brown-out recovery)
                    recharge_attempts = 0
                    while self.cap.get_current_voltage() < 3.0 and recharge_attempts < 1000:
                        self.harvester.step_harvest(self.cap, 10.0)
                        self.total_cycles += 160000 # Time spent charging
                        recharge_attempts += 1

                    if recharge_attempts >= 1000:
                        if epoch == epochs - 1:
                            print(f" [!] Persistent Blackout in Epoch {epoch}.")
                        break

                # 5. Data Logging (Matches main.py expectations)
                self.trace_logs.append({
                    "Epoch": epoch,
                    "PC_Index": pc,
                    "Source_Line": original_line_no,
                    "Voltage": v_after,
                    "Inflow": inflow,
                    "Action": action,
                    "Status": status,
                    "Total_Cycles": self.total_cycles  # Column name must match main.py
                })

                # 6. Agent Learning Update
                self.agent.learn(
                    v_before, pc_percent, inflow, action, status, cycle_cost,
                    v_after, (pc / len(code_trace)) * 100, inflow
                )

            # --- FINAL EPOCH: PHASE-STYLE LOGGING ---
            if epoch == epochs - 1:
                print("\n" + "█" * 85)
                print(f"  ML_ADAPTIVE: DYNAMIC CHECKPOINT TRACE (Scenario: {self.harvester.scenario})")
                print("█" * 85)

                if not epoch_checkpoints:
                    print("  [No Checkpoints Placed - Optimal Execution]")
                else:
                    for i, cp in enumerate(epoch_checkpoints):
                        # Format code snippet to fit cleanly in terminal
                        snippet = (cp['content'][:45] + '..') if len(cp['content']) > 45 else cp['content']
                        print(f"  CP {i + 1:02}: [Line {cp['line']:<3}] -> {snippet}")

                print(f"\n  Total Checkpoints: {len(epoch_checkpoints)} | Final Cycles: {self.total_cycles}")
                print("█" * 85 + "\n")

        # 7. Finalize and Save Telemetry
        filename = f"output_trace_{self.harvester.scenario.lower()}.csv"
        self.save_trace(filename)

    def save_trace(self, filename):
        """Saves simulation telemetry to CSV for post-simulation analysis."""
        if not self.trace_logs:
            df = pd.DataFrame([{"Status": "FAILED_BLACKOUT"}])
        else:
            df = pd.DataFrame(self.trace_logs)

        df.to_csv(filename, index=False)
        print(f"Full hardware trace saved to {filename}")