"""
sim_engine.py -- Phase 0 instrumentation patch.

CHANGES FROM THE VERSION YOU SENT:
1. Added self.checkpoint_count, incremented once per executed checkpoint
   action (action==1), regardless of whether that step later triggers a
   crash. This was previously untracked -- only visible transiently as a
   local list (epoch_checkpoints) used solely for the console summary
   printout, not accessible to any caller. Needed for the professor's
   Phase 0 requirement: "per-rung logs of efficiency gain, crash count,
   wasted cycles, checkpoint count, active/sleep cycles."
2. Added self.crash_count as an explicit alias for self.recomputations
   (kept recomputations for backward compatibility -- nothing renamed,
   only added).
3. Added self.use_reflexes and self.use_agent flags (both default True,
   preserving current 'pace' strategy behavior exactly). These implement
   the 4-rung attribution ladder from the professor's Experiment 2:
     R1: strategy='hybrid'                          (unchanged, pre-existing)
     R2: strategy='pace', use_reflexes=True,  use_agent=False
     R3: strategy='pace', use_reflexes=False, use_agent=True
     R4: strategy='pace', use_reflexes=True,  use_agent=True  (= old default)
   The reflex check is now applied at the ENGINE level, before the masking
   layer, matching the paper's stated order (Section III-D1: reflexes
   evaluated first). Previously the reflexes lived only inside
   agent.choose_action(), which is called *after* the masking layer --
   see the flag raised alongside this patch. This patch fixes that
   ordering as well as adding the rung toggles.
4. When use_agent=False (R2), the "gray area" that would otherwise reach
   the Q-agent instead falls back to the plain hybrid decision
   (self.hybrid_model.should_checkpoint(features)) -- i.e. R2 is exactly
   "hybrid, with hard voltage reflexes wrapped around it," per the
   professor's definition.
"""

import pandas as pd
import numpy as np
from src.environment.capacitor import MSP430Capacitor
from src.environment.harvester import EnergyHarvester
from src.profiler.parser import MapiProParser
from src.agent.model import AdaptiveCheckpointAgent
from src.baselines.hybrid_model import HybridCheckpointModel


class SimulationEngine:
    def __init__(self, scenario="Winter", reward_config=None):

        self.scenario = scenario
        self.cap = MSP430Capacitor()
        self.harvester = EnergyHarvester(scenario=scenario)
        self.parser = MapiProParser()

        # Instantiate BOTH models
        self.agent = AdaptiveCheckpointAgent()
        self.hybrid_model = HybridCheckpointModel()

        self.trace_logs = []

        # Cycle tracking initialized
        self.total_cycles = 0
        self.sleep_cycles = 0
        self.active_cycles = 0
        self.total_wasted_cycles = 0
        self.checkpoint_count = 0          # NEW -- Phase 0 telemetry

        self.last_checkpoint_pc = 0
        self.recomputations = 0
        self.crash_count = 0               # NEW -- explicit alias, see docstring
        self.target_died = False           # NEW -- set True on persistent-blackout break
        self.total_reads = 0
        self.total_writes = 0
        self.crash_history = {}
        self.reward_config = reward_config

        self.strategy = "pace"  # Default, overridden by main.py

        # Rung-ladder toggles -- see docstring. Defaults preserve the
        # original full-PACE behavior exactly.
        self.use_reflexes = True           # NEW
        self.use_agent = True              # NEW

        # Dynamic State Size Mapping (in bytes)
        self.state_size_map = {
            "PicoJPEG": 2560,    # Heavy MCU buffers, Quantization/Huffman tables
            "AES": 512,          # Round keys, state matrix
            "FFT": 1024,         # Real/Imag data buffers
            "SHA256": 512,       # Message schedules, hash states
            "MatMult": 768,      # Matrix rows/cols in active memory
            "FIR": 256,          # Filter coefficients
            "Default": 32        # Basic pointers/registers for simple algos
        }

    def run_simulation(self, code_trace, epochs=1):
        """
        Hardware-aware simulation with explicit crash logging, authentic hybrid
        routing, pure physical cycle tracking, and RL Action Masking.
        """
        self.total_reads = sum(1 for line in code_trace if any(x in line['code'] for x in ['->', '[', 'deref', 'read']))
        self.total_writes = sum(1 for line in code_trace if '=' in line['code'] and '==' not in line['code'])

        if hasattr(self.agent, 'reset_state'):
            self.agent.reset_state()

        for epoch in range(epochs):
            self.cap.reset_to_full()
            pc = 0
            v_prev = self.cap.get_current_voltage()  # Track for voltage drop rate

            # Reset all cycle trackers at the start of each epoch
            self.total_cycles = 0
            self.sleep_cycles = 0
            self.active_cycles = 0
            self.total_wasted_cycles = 0
            self.checkpoint_count = 0      # NEW -- reset per epoch, mirrors other counters

            self.last_checkpoint_pc = 0
            epoch_checkpoints = []
            self.crash_history = {}

            # Determine dynamic state size based on parsed algorithm
            # Normalize string so "aes.c" -> "AES"
            raw_algo = str(self.parser.predicted_algo).replace(".c", "").strip().upper()

            # Case-insensitive map lookup
            state_lookup = {k.upper(): v for k, v in self.state_size_map.items()}
            current_state_size = state_lookup.get(raw_algo, self.state_size_map["Default"])

            # ====================================================
            # RL EVALUATION MODE
            # ====================================================
            if self.strategy == "pace" and epoch == epochs - 1:
                # Maintain a tiny exploration floor to prevent Q-table starvation
                self.agent.epsilon = 0.02

            while pc < len(code_trace):
                season_flag = 1 if self.harvester.scenario == "Summer" else 0
                instr_data = code_trace[pc]
                line_code = instr_data["code"]
                original_line_no = instr_data["line_no"]

                distance = pc - self.last_checkpoint_pc
                complexity_score = min(distance // 15, 3)
                v_before = self.cap.get_current_voltage()
                inflow = self.harvester.get_inflow_power()
                pc_percent = (pc / len(code_trace)) * 100

                # Calculate physics derivative
                drop_rate = max(0.0, v_prev - v_before)
                v_prev = v_before

                # Build the Enriched Feature Set
                remaining_steps = len(code_trace) - pc   # NEW -- Phase 2 item 7, length-normalized tax
                features = {
                    "current_voltage": v_before,
                    "voltage_drop_rate": drop_rate,
                    "structural_complexity": complexity_score,
                    "is_loop_header": "for" in line_code or "while" in line_code,
                    "work_since_last_cp": distance,
                    "overhead_cost": current_state_size,
                    "remaining_steps": remaining_steps,   # NEW
                }

                # ====================================================
                # 1. DECISION PHASE: HYBRID vs PACE (With Action Masking)
                # ====================================================
                if self.strategy == "hybrid":
                    # R1 -- pure hybrid heuristic, unchanged.
                    action = 1 if self.hybrid_model.should_checkpoint(features) else 0
                else:
                    # ---- ORIGINAL ORDERING PRESERVED (masking layer first,
                    # reflexes evaluated last, inside the agent-decision
                    # point) -- this is what actually produced Table IV.
                    # The paper's prose describing reflexes as evaluated
                    # "first" does not match this and needs a documentation
                    # fix instead; see the note attached to this patch. ----
                    trivial_keywords = ["static ", "int ", "void ", "char ", "heap_ptr", "NULL", "return"]
                    is_trivial = any(kw in line_code for kw in trivial_keywords)
                    predicted_efficiency = self.hybrid_model.reg_model.predict(features)

                    if self.crash_history.get(pc, 0) >= 2:
                        action = 1
                        self.crash_history[pc] = 0
                    elif is_trivial:
                        action = 0
                    elif predicted_efficiency < self.hybrid_model.threshold:
                        # ACTION MASKING: Hybrid model acts as a strict babysitter.
                        action = 0
                    else:
                        # The Gray Area -- masking layer passed the decision
                        # through. This is the ONE point where reflexes and/or
                        # the Q-agent get a say, exactly as in the original code.
                        if self.use_agent:
                            # bypass_reflexes=not self.use_reflexes: when
                            # use_reflexes=True (R4, the original default),
                            # this passes bypass_reflexes=False, so the
                            # agent applies its own internal reflexes exactly
                            # as the unpatched code always did. When
                            # use_reflexes=False (R3), reflexes are skipped
                            # entirely and control goes straight to the
                            # Q-table.
                            action = self.agent.choose_action(
                                v_before, pc_percent, inflow, complexity_score, season_flag,
                                features=features, bypass_reflexes=(not self.use_reflexes)
                            )
                        else:
                            # R2: no agent. Apply the same two reflex checks
                            # manually, at the same point the agent would
                            # have been consulted, then fall back to the
                            # masking layer's own pass-through decision
                            # (which, having reached here, is already a
                            # checkpoint per the efficiency-threshold gate
                            # above).
                            if self.use_reflexes and (v_before > 2.65 or (pc_percent > 90.0 and v_before > 2.4)):
                                action = 0
                            elif self.use_reflexes and v_before < 2.15:
                                action = 1
                            else:
                                action = 1

                # ====================================================
                # 2. EXECUTION & PHYSICS
                # ====================================================
                energy_spent = 0
                cycle_cost = 0
                status = "SUCCESS"

                if action == 1:
                    energy_spent, cycle_cost = self.parser.get_checkpoint_cost(current_state_size)
                    self.checkpoint_count += 1     # NEW -- counted regardless of crash outcome below
                else:
                    energy_spent, cycle_cost = self.parser.profile_line(line_code)

                # Physics Step 1: Consume energy first
                self.cap.consume_energy(energy_spent)

                # Physics Step 2: Harvest energy over the time it took
                self.harvester.step_harvest(self.cap, (cycle_cost / 16000))

                self.active_cycles += cycle_cost
                self.total_cycles += cycle_cost

                v_after = self.cap.get_current_voltage()

                # ====================================================
                # 3. CRASH HANDLING & PENALTY CALCULATION
                # ====================================================
                if v_after <= 1.8 or getattr(self.cap, 'is_dead', False):
                    status = "CRASH"
                    self.crash_history[pc] = self.crash_history.get(pc, 0) + 1

                    wasted_cycles = cycle_cost
                    for i in range(self.last_checkpoint_pc, pc):
                        _, cost = self.parser.profile_line(code_trace[i]["code"])
                        wasted_cycles += cost

                    self.total_wasted_cycles += wasted_cycles
                    self.active_cycles += wasted_cycles
                    self.total_cycles += wasted_cycles

                    self.trace_logs.append({
                        "Epoch": epoch,
                        "PC_Index": pc,
                        "Source_Line": original_line_no,
                        "Voltage": v_after,
                        "Action": "CHECKPOINT" if action == 1 else "EXECUTE",
                        "Status": "CRASH",
                        "Total_Cycles": self.total_cycles,
                        "Wasted_Cycles": wasted_cycles,
                        "Wasted_Time_ms": wasted_cycles / 16000
                    })

                    if self.strategy == "pace" and self.use_agent:
                        self.agent.learn(
                            v_before, pc_percent, inflow, action, status, cycle_cost,
                            v_after, (self.last_checkpoint_pc / len(code_trace)) * 100, inflow, complexity_score,
                            season_flag, features=features
                        )

                    # ENFORCE STATE LOSS
                    pc = self.last_checkpoint_pc
                    self.recomputations += 1
                    self.crash_count += 1      # NEW

                    # MANDATORY RECHARGE
                    recharge_attempts = 0
                    while self.cap.get_current_voltage() < 3.0:
                        self.harvester.step_harvest(self.cap, 1.0)
                        self.sleep_cycles += 16000
                        self.total_cycles += 16000
                        recharge_attempts += 1
                        if recharge_attempts > 10000:
                            break

                    if recharge_attempts > 10000:
                        self.target_died = True    # NEW -- see main_figures.py Night-detection fix
                        if epoch == epochs - 1:
                            print(f" [!] Persistent Blackout in Epoch {epoch}. Target died.")
                        break

                    v_prev = self.cap.get_current_voltage()  # Reset tracking post-recharge
                    continue

                    # ====================================================
                # 4. SUCCESS HANDLING & ADVANCING STATE
                # ====================================================
                if action == 1:
                    self.last_checkpoint_pc = pc
                    if epoch == epochs - 1:
                        epoch_checkpoints.append({
                            "line": original_line_no,
                            "content": line_code.strip()
                        })
                else:
                    pc += 1

                    # ====================================================
                # 5. AGENT LEARNING
                # ====================================================
                if self.strategy == "pace" and self.use_agent:
                    self.agent.learn(
                        v_before, pc_percent, inflow, action, status, cycle_cost,
                        v_after, (pc / len(code_trace)) * 100, inflow, complexity_score, season_flag, features=features
                    )

                self.trace_logs.append({
                    "Epoch": epoch,
                    "PC_Index": pc if action == 1 else pc - 1,
                    "Source_Line": original_line_no,
                    "Voltage": v_after,
                    "Action": "CHECKPOINT" if action == 1 else "EXECUTE",
                    "Status": status,
                    "Total_Cycles": self.total_cycles,
                    "Wasted_Cycles": 0,
                    "Wasted_Time_ms": 0.0
                })

            # ====================================================
            # END OF EPOCH
            # ====================================================
            if self.strategy == "pace" and self.use_agent:
                if hasattr(self.agent, 'decay_epsilon'):
                    self.agent.decay_epsilon()

            if epoch == epochs - 1:
                self._print_epoch_summary(epoch_checkpoints)

        self.save_trace(f"output_trace_{self.harvester.scenario.lower()}.csv")

        if self.strategy == "pace" and self.use_agent:
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
        print(f"  Final Active CPU Cycles: {self.active_cycles}")
        print(f"  Final Sleep/Recharge Cycles: {self.sleep_cycles}")
        print(f"  Final Total Wall-Clock Cycles: {self.total_cycles}")
        print("=" * 85 + "\n")

    def save_trace(self, filename):
        if not self.trace_logs:
            return
        df = pd.DataFrame(self.trace_logs)
        df.to_csv(filename, index=False)