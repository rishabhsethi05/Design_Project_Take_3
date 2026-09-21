"""
src/experiment_harness.py

Phase 0 deliverable: a single, correctly-seeded harness that every
experiment script (tuner.py, generalization experiments, the attribution
ladder) should call through, instead of each script reimplementing its
own train/eval loop with its own seeding order.

WHY THIS FILE EXISTS (read this before using it)
--------------------------------------------------
We found that main_figures.py produced numbers that didn't match
tuner.py for identical (benchmark, scenario, safe_v, tax, seeds) inputs
-- in several cases the sign of the reported gain flipped entirely.

Root cause: main_figures.py's run_experiment() does, per seed:
    set_seed(s) -> run HYBRID baseline -> run PACE training -> run PACE eval
The hybrid run consumes random draws from the harvester's stochastic
Winter/SPF model. That shifts the random-number stream's position before
PACE training even starts, so PACE sees a *different* effective random
sequence than it would if trained right after set_seed(s) with nothing
in between -- which is what tuner.py's evaluate_candidate() does (hybrid
baselines are computed in a completely separate loop, earlier, before
tuning starts at all).

Since harvester stochasticity directly controls *when crashes happen*,
and PACE's behavior is often boundary-sensitive near a crash threshold,
even a small stream-offset shift can cascade into a completely different
crash pattern -- consistent with what we observed (large benchmarks like
aes drifted slightly; boundary-sensitive ones like fir/crc/stringsearch1
flipped sign).

THE RULE THIS FILE ENFORCES: every independent "thing that consumes
randomness" (one hybrid run, one PACE training+eval run, one rung
evaluation) gets its own fresh set_seed(s) call immediately before it,
with nothing else touching the RNG in between. Never interleave two
different stochastic runs inside one seeded block.
"""

import os
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.controller.sim_engine import SimulationEngine
from src.agent.model import AdaptiveCheckpointAgent


# ============================================================
# Seeding
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================
# Workload loading (cached per benchmark -- avoids re-parsing the same
# .c file repeatedly; the *content* returned is still safe to reuse
# across runs since nothing mutates it in place -- run_simulation reads
# code_trace, it never writes to it)
# ============================================================
_workload_cache = {}


def get_workload(benchmark_file, multiplier=5, base_path=None):
    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "benchmarks")
    key = (benchmark_file, multiplier, base_path)
    if key not in _workload_cache:
        benchmark_path = os.path.join(base_path, benchmark_file)
        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Missing benchmark: {benchmark_path}")
        engine = SimulationEngine(scenario="Summer")  # scenario irrelevant for parsing
        code = engine.parser.load_c_file(benchmark_path)
        _workload_cache[key] = code * multiplier
    return _workload_cache[key]


# ============================================================
# Telemetry container -- one of these per (benchmark, scenario, rung, seed)
# or per averaged-across-seeds result. Matches the professor's Phase 0
# requirement: efficiency gain is computed by the caller (needs a
# baseline to compare against); everything else is captured here.
# ============================================================
@dataclass
class RunTelemetry:
    total_cycles: int = 0
    active_cycles: int = 0
    sleep_cycles: int = 0
    wasted_cycles: int = 0
    checkpoint_count: int = 0
    crash_count: int = 0
    target_died: bool = False
    valid: bool = True  # False if this run should be excluded (e.g. cycles <= 0)

    @classmethod
    def from_engine(cls, sim: SimulationEngine):
        cycles = getattr(sim, 'total_cycles', 0)
        return cls(
            total_cycles=cycles,
            active_cycles=getattr(sim, 'active_cycles', 0),
            sleep_cycles=getattr(sim, 'sleep_cycles', 0),
            wasted_cycles=getattr(sim, 'total_wasted_cycles', 0),
            checkpoint_count=getattr(sim, 'checkpoint_count', 0),
            crash_count=getattr(sim, 'crash_count', 0),
            target_died=getattr(sim, 'target_died', False),
            valid=(cycles > 0),
        )


MIN_VALID_CYCLES = 2000


# ============================================================
# HYBRID BASELINE (R1) -- one clean, independently-seeded run.
# ============================================================
def run_hybrid_once(benchmark_file, scenario, seed, multiplier=5, epochs=1) -> RunTelemetry:
    set_seed(seed)  # nothing else touches the RNG before this run -- see module docstring
    workload = get_workload(benchmark_file, multiplier)

    sim = SimulationEngine(scenario=scenario)
    sim.strategy = 'hybrid'
    sim.parser.predicted_algo = benchmark_file
    sim._print_epoch_summary = lambda *a, **k: None
    sim.run_simulation(workload, epochs=epochs)

    return RunTelemetry.from_engine(sim)


# ============================================================
# TRAIN a PACE agent for one rung, one seed. Returns the trained agent
# (its Q-table is the "policy" -- keep it if you need to freeze and
# reuse it later, e.g. for Experiment 1's cross-scenario freeze).
# ============================================================
def train_pace_agent(benchmark_file, scenario, seed, safe_v, tax, epochs, multiplier=5,
                      use_reflexes=True, use_agent=True) -> AdaptiveCheckpointAgent:
    set_seed(seed)  # independently seeded -- nothing else must run before this in the same block
    workload = get_workload(benchmark_file, multiplier)
    estimated_steps = len(workload) * epochs

    agent = AdaptiveCheckpointAgent(alpha=0.10, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01)
    agent.safe_v_threshold = safe_v
    agent.base_spam_tax = tax
    agent.epsilon_decay = (agent.epsilon_min / agent.epsilon_start) ** (1.0 / max(500, int(estimated_steps * 0.75)))

    sim = SimulationEngine(scenario=scenario)
    sim.strategy = 'pace'
    sim.agent = agent
    sim.use_reflexes = use_reflexes
    sim.use_agent = use_agent
    sim.parser.predicted_algo = benchmark_file
    sim._print_epoch_summary = lambda *a, **k: None
    sim.run_simulation(workload, epochs=epochs)

    return agent


# ============================================================
# EVALUATE a (possibly frozen) PACE agent for one rung, one seed.
#
# freeze=True implements "no Q-updates" WITHOUT touching sim_engine.py's
# learning call sites: setting alpha=0 makes the Bellman update a no-op
# (Q <- (1-0)*Q + 0*(...) == Q unchanged) while everything else
# (steps_since_cp bookkeeping, epsilon=0 greedy action selection) still
# runs exactly as it would in real deployment. This is what makes
# "freeze everything, change nothing" (Experiment 1) actually true
# rather than aspirational -- alpha is restored afterward so a frozen
# evaluation never permanently mutates an agent you plan to reuse.
# ============================================================
def evaluate_pace_agent(benchmark_file, scenario, agent: AdaptiveCheckpointAgent, multiplier=5,
                         use_reflexes=True, use_agent=True, freeze=True, reseed=None) -> RunTelemetry:
    """
    reseed: controls RNG behavior relative to whatever run happened
    immediately before this call.

      reseed=None (default) -- do NOT call set_seed(). Evaluation
      continues from wherever the RNG stream currently is. This matches
      tuner.py's original convention exactly: one set_seed(s) call per
      seed, train and eval share that one continuous stream. USE THIS
      for in-sample replication (Experiment 2's R2/R3/R4 rungs, or
      anything meant to reproduce Table IV numbers).

      reseed=<int> -- explicitly call set_seed(<int>) before evaluating.
      USE THIS for genuinely independent evaluation: Experiment 1's
      cross-scenario freeze (train on Summer seed s, evaluate the frozen
      policy on Winter under its own seed) or cross-trace held-out-seed
      evaluation (evaluate under a seed never used in training).
    """
    if reseed is not None:
        set_seed(reseed)
    workload = get_workload(benchmark_file, multiplier)

    agent.reset_state()
    agent.epsilon = 0.0
    original_alpha = agent.alpha
    if freeze:
        agent.alpha = 0.0

    try:
        sim = SimulationEngine(scenario=scenario)
        sim.strategy = 'pace'
        sim.agent = agent
        sim.use_reflexes = use_reflexes
        sim.use_agent = use_agent
        sim.parser.predicted_algo = benchmark_file
        sim._print_epoch_summary = lambda *a, **k: None
        sim.run_simulation(workload, epochs=1)
        telemetry = RunTelemetry.from_engine(sim)
    finally:
        agent.alpha = original_alpha  # never leave the agent permanently mutated

    return telemetry


# ============================================================
# Convenience: run one full (train + evaluate) cycle for one rung, one
# seed. This is the R2/R3/R4 building block for Experiment 2. R1 doesn't
# need training -- use run_hybrid_once directly.
# ============================================================
def run_rung_once(benchmark_file, scenario, seed, safe_v, tax, epochs, multiplier=5,
                   use_reflexes=True, use_agent=True) -> RunTelemetry:
    """
    In-sample replication path (matches tuner.py exactly): train and
    evaluate share ONE continuous RNG stream from a single set_seed(seed)
    call -- evaluate_pace_agent is called with reseed=None on purpose.
    This is what Experiment 2's R2/R3/R4 rungs should use.
    For Experiment 1 (held-out / cross-scenario), do NOT use this
    function -- call train_pace_agent and evaluate_pace_agent separately
    with an explicit reseed= value for evaluation instead.
    """
    if use_agent:
        agent = train_pace_agent(
            benchmark_file, scenario, seed, safe_v, tax, epochs, multiplier,
            use_reflexes=use_reflexes, use_agent=use_agent
        )
        return evaluate_pace_agent(
            benchmark_file, scenario, agent, multiplier,
            use_reflexes=use_reflexes, use_agent=use_agent, freeze=True, reseed=None
        )
    else:
        # R2: no agent, so "training" is meaningless -- reflexes+hybrid
        # only, evaluated directly. Still gets its own clean seed.
        set_seed(seed)
        workload = get_workload(benchmark_file, multiplier)
        dummy_agent = AdaptiveCheckpointAgent()  # never consulted when use_agent=False
        sim = SimulationEngine(scenario=scenario)
        sim.strategy = 'pace'
        sim.agent = dummy_agent
        sim.use_reflexes = use_reflexes
        sim.use_agent = False
        sim.parser.predicted_algo = benchmark_file
        sim._print_epoch_summary = lambda *a, **k: None
        sim.run_simulation(workload, epochs=1)
        return RunTelemetry.from_engine(sim)


# ============================================================
# Averaging across seeds, with the survival/validity rule shared by
# tuner.py and every experiment script (kept identical on purpose).
# ============================================================
def average_telemetry(telemetries: list[RunTelemetry]) -> Optional[RunTelemetry]:
    valid = [t for t in telemetries if t.valid and t.total_cycles >= MIN_VALID_CYCLES]
    if not valid:
        return None
    n = len(valid)
    return RunTelemetry(
        total_cycles=int(np.mean([t.total_cycles for t in valid])),
        active_cycles=int(np.mean([t.active_cycles for t in valid])),
        sleep_cycles=int(np.mean([t.sleep_cycles for t in valid])),
        wasted_cycles=int(np.mean([t.wasted_cycles for t in valid])),
        checkpoint_count=int(np.mean([t.checkpoint_count for t in valid])),
        crash_count=int(np.mean([t.crash_count for t in valid])),
        target_died=any(t.target_died for t in valid),
        valid=(n == len(telemetries)),  # False if any seed was dropped
    )


def compute_gain(baseline_cycles: int, pace_cycles: int) -> Optional[float]:
    if baseline_cycles < MIN_VALID_CYCLES or pace_cycles < MIN_VALID_CYCLES:
        return None
    return ((baseline_cycles - pace_cycles) / baseline_cycles) * 100.0