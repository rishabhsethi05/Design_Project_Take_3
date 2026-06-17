import pandas as pd
import numpy as np
from src.environment.capacitor import MSP430Capacitor
from src.environment.harvester import EnergyHarvester
from src.profiler.parser import MapiProParser
from src.agent.model import AdaptiveCheckpointAgent

# Import the authentic Hybrid model we built earlier
from src.baselines.hybrid_model import HybridCheckpointModel


class SimulationEngine:
    def __init__(self, scenario="Winter", reward_config=None):

        self.scenario = scenario
        self.cap = MSP430Capacitor()
        self.harvester = EnergyHarvester(scenario=scenario)
        self.parser = MapiProParser()

        # Instantiate BOTH models so we can test them against each other
        self.agent = AdaptiveCheckpointAgent()
        self.hybrid_model = HybridCheckpointModel()

        self.trace_logs = []
        self.total_cycles = 0
        self.last_checkpoint_pc = 0
        self.recomputations = 0

        self.total_wasted_cycles = 0
        self.total_reads = 0
        self.total_writes = 0
        self.crash_history = {}
        self.reward_config = reward_config


        self.strategy = "pace"  # Default, overridden by main.py

    def run_simulation(self, code_trace, epochs=1):
        """
        Hardware-aware simulation with explicit crash logging, authentic hybrid
        routing, and pure physical cycle tracking.
        """
        self.total_reads = sum(1 for line in code_trace if any(x in line['code'] for x in ['->', '[', 'deref', 'read']))
        self.total_writes = sum(1 for line in code_trace if '=' in line['code'] and '==' not in line['code'])

        for epoch in range(epochs):
            self.cap.reset_to_full()
            pc = 0
            self.total_cycles = 0
            self.last_checkpoint_pc = 0
            self.total_wasted_cycles = 0
            epoch_checkpoints = []

            # ====================================================
            # RL EVALUATION MODE
            # ====================================================
            if self.strategy == "pace" and epoch == epochs - 1:
                self.agent.epsilon = 0.0

            while pc < len(code_trace):
                instr_data = code_trace[pc]
                line_code = instr_data["code"]
                original_line_no = instr_data["line_no"]

                complexity_score = self.parser.get_cyclomatic_complexity(line_code)
                v_before = self.cap.get_current_voltage()
                inflow = self.harvester.get_inflow_power()
                pc_percent = (pc / len(code_trace)) * 100

                # ====================================================
                # 1. DECISION PHASE: HYBRID vs PACE
                # ====================================================
                if self.strategy == "hybrid":
                    features = {
                        "current_voltage": v_before,
                        "voltage_drop_rate": 0.05,
                        "structural_complexity": complexity_score,
                        "is_loop_header": "for" in line_code or "while" in line_code,
                        "work_since_last_cp": pc - self.last_checkpoint_pc,
                        "overhead_cost": 32
                    }
                    action = 1 if self.hybrid_model.should_checkpoint(features) else 0
                else:
                    # THE WATCHDOG: If we crashed here twice, FORCE a save.
                    if self.crash_history.get(pc, 0) >= 2:
                        action = 1
                        self.crash_history[pc] = 0  # Reset watchdog
                    # TRUE PHYSICAL GUARDRAIL: Minimum Forward Progress Cooldown
                    # The agent must execute at least 3 instructions before saving again.
                    elif (pc - self.last_checkpoint_pc) < 3:
                        action = 0
                    else:
                        action = self.agent.choose_action(v_before, pc_percent, inflow, complexity_score)

                # ====================================================
                # 2. EXECUTION & PHYSICS
                # ====================================================
                energy_spent = 0
                cycle_cost = 0
                status = "SUCCESS"

                if action == 1:
                    energy_spent, cycle_cost = self.parser.get_checkpoint_cost(32)
                    self.last_checkpoint_pc = pc
                    if epoch == epochs - 1:
                        epoch_checkpoints.append({
                            "line": original_line_no,
                            "content": line_code.strip()
                        })
                else:
                    energy_spent, cycle_cost = self.parser.profile_line(line_code)
                    pc += 1

                self.cap.consume_energy(energy_spent)
                self.harvester.step_harvest(self.cap, (cycle_cost / 16000000))
                self.total_cycles += cycle_cost
                v_after = self.cap.get_current_voltage()

                # ====================================================
                # 3. CRASH HANDLING & PENALTY CALCULATION
                # ====================================================
                if self.cap.is_dead:
                    status = "CRASH"
                    self.crash_history[pc] = self.crash_history.get(pc, 0) + 1  # Log the exact crash site

                    wasted_cycles = 0
                    for i in range(self.last_checkpoint_pc, pc):
                        _, cost = self.parser.profile_line(code_trace[i]["code"])
                        wasted_cycles += cost

                    self.total_wasted_cycles += wasted_cycles
                    self.total_cycles += wasted_cycles

                    self.trace_logs.append({
                        "Epoch": epoch,
                        "PC_Index": pc,
                        "Source_Line": original_line_no,
                        "Voltage": v_after,
                        "Action": "EXECUTE",
                        "Status": "CRASH",
                        "Total_Cycles": self.total_cycles,
                        "Wasted_Cycles": wasted_cycles,
                        "Wasted_Time_ms": wasted_cycles / 16000
                    })

                    if self.strategy == "pace":
                        self.agent.learn(
                            v_before, pc_percent, inflow, action, status, cycle_cost,
                            v_after, (self.last_checkpoint_pc / len(code_trace)) * 100, inflow, complexity_score
                        )

                    pc = self.last_checkpoint_pc
                    self.recomputations += 1

                    recharge_attempts = 0
                    while self.cap.get_current_voltage() < 3.0:
                        self.harvester.step_harvest(self.cap, 1.0)
                        self.total_cycles += 16000
                        recharge_attempts += 1

                        if recharge_attempts > 10000:
                            break

                    if recharge_attempts > 10000:
                        if epoch == epochs - 1:
                            print(f" [!] Persistent Blackout in Epoch {epoch}. Target died.")
                        break

                    continue

                # ====================================================
                # 4. AGENT LEARNING
                # ====================================================
                if self.strategy == "pace":
                    self.agent.learn(
                        v_before, pc_percent, inflow, action, status, cycle_cost,
                        v_after, (pc / len(code_trace)) * 100, inflow, complexity_score
                    )

                self.trace_logs.append({
                    "Epoch": epoch,
                    "PC_Index": pc,
                    "Source_Line": original_line_no,
                    "Voltage": v_after,
                    "Action": "CHECKPOINT" if action == 1 else "EXECUTE",
                    "Status": status,
                    "Total_Cycles": self.total_cycles,
                    "Wasted_Cycles": 0,
                    "Wasted_Time_ms": 0.0
                })

            if epoch == epochs - 1:
                self._print_epoch_summary(epoch_checkpoints)

        self.save_trace(f"output_trace_{self.harvester.scenario.lower()}.csv")

        if self.strategy == "pace":
            self.agent.save_model("mapi_pro_agent.pkl")

    def _print_epoch_summary(self, checkpoints):
        exec_time_seconds = self.total_cycles / 16000000

        print("\n" + "=" * 85)
        print(f"  {self.strategy.upper()}: DYNAMIC CHECKPOINT TRACE (Scenario: {self.harvester.scenario})")
        print("=" * 85)
        print(f"  [Memory Access Profile]")
        print(f"  - Total Memory Reads:   {self.total_reads}")
        print(f"  - Total Memory Writes:  {self.total_writes}")
        print(f"  [Temporal Performance]")
        print(f"  - Total Execution Time: {exec_time_seconds:.6f} seconds")
        print("-" * 85)

        if not checkpoints:
            print("  [No Checkpoints Placed - High Risk Execution]")
        else:
            for i, cp in enumerate(checkpoints[:40]):
                snippet = (cp['content'][:45] + '..') if len(cp['content']) > 45 else cp['content']
                print(f"  CP {i + 1:02}: [Line {cp['line']:<3}] -> {snippet}")

        print(f"\n  Total Checkpoints: {len(checkpoints)}")
        print(f"  Total Recomputed/Wasted cycles: {self.total_wasted_cycles}")
        print(f"  Final Cycles: {self.total_cycles}")
        print("=" * 85 + "\n")

    def save_trace(self, filename):
        if not self.trace_logs:
            return
        df = pd.DataFrame(self.trace_logs)
        df.to_csv(filename, index=False)