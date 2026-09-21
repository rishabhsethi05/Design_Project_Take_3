import sys
sys.path.insert(0, ".")

from src.experiment_harness import (
    run_hybrid_once, evaluate_pace_agent, average_telemetry,
    compute_gain, get_workload, set_seed
)
from src.agent.model import AdaptiveCheckpointAgent
from src.controller.sim_engine import SimulationEngine

TRAIN_SEEDS = [42, 123, 999]
EPOCHS = 15
MULTIPLIER = 5
SCENARIO = "Winter"

SIX_FRAGILE = {
    "bs.c":        {"safe_v": 2.00, "tax": 0.00},
    "cnt.c":       {"safe_v": 2.00, "tax": 0.50},
    "cubic.c":     {"safe_v": 1.90, "tax": 0.25},
    "fibcall.c":   {"safe_v": 2.00, "tax": 0.00},
    "recursion.c": {"safe_v": 1.90, "tax": 0.00},
    "select.c":    {"safe_v": 1.90, "tax": 0.00},
}


def train_with_flag(bench, seed, safe_v, tax, use_length_norm):
    set_seed(seed)
    workload = get_workload(bench, MULTIPLIER)
    estimated_steps = len(workload) * EPOCHS
    agent = AdaptiveCheckpointAgent(alpha=0.10, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01)
    agent.safe_v_threshold = safe_v
    agent.base_spam_tax = tax
    agent.use_length_normalized_tax = use_length_norm
    agent.epsilon_decay = (agent.epsilon_min / agent.epsilon_start) ** (1.0 / max(500, int(estimated_steps * 0.75)))
    sim = SimulationEngine(scenario=SCENARIO)
    sim.strategy = 'pace'
    sim.agent = agent
    sim.parser.predicted_algo = bench
    sim._print_epoch_summary = lambda *a, **k: None
    sim.run_simulation(workload, epochs=EPOCHS)
    return agent


def run_condition(bench, safe_v, tax, use_length_norm):
    hybrid_runs = [run_hybrid_once(bench, SCENARIO, s, multiplier=MULTIPLIER, epochs=1) for s in TRAIN_SEEDS]
    avg_hybrid = average_telemetry(hybrid_runs)

    pace_runs = []
    for s in TRAIN_SEEDS:
        agent = train_with_flag(bench, s, safe_v, tax, use_length_norm)
        telem = evaluate_pace_agent(bench, SCENARIO, agent, multiplier=MULTIPLIER, freeze=False, reseed=None)
        pace_runs.append(telem)

    avg_pace = average_telemetry(pace_runs)
    if not (avg_hybrid and avg_pace):
        return None
    return {
        "gain": compute_gain(avg_hybrid.total_cycles, avg_pace.total_cycles),
        "checkpoint_count": avg_pace.checkpoint_count,
        "crash_count": avg_pace.crash_count,
        "wasted_cycles": avg_pace.wasted_cycles,
    }


if __name__ == "__main__":
    print(f"{'='*100}\nPHASE 2 ITEM 7 -- Length-Normalized Tax: Before/After on the Six Fragile Benchmarks\n{'='*100}\n")

    for bench, prof in SIX_FRAGILE.items():
        trace_len = len(get_workload(bench, MULTIPLIER))
        print(f"\n{'-'*100}\n{bench}  (trace length = {trace_len} steps, tax_reference_length default = 1000)\n{'-'*100}")

        before = run_condition(bench, prof["safe_v"], prof["tax"], use_length_norm=False)
        after = run_condition(bench, prof["safe_v"], prof["tax"], use_length_norm=True)

        print(f"  STANDARD tax   : gain={before['gain']:7.2f}%  checkpoints={before['checkpoint_count']}  "
              f"crashes={before['crash_count']}  wasted={before['wasted_cycles']:,}")
        print(f"  NORMALIZED tax : gain={after['gain']:7.2f}%  checkpoints={after['checkpoint_count']}  "
              f"crashes={after['crash_count']}  wasted={after['wasted_cycles']:,}")
        delta = after['gain'] - before['gain']
        verdict = "IMPROVED" if delta > 1.0 else ("WORSE" if delta < -1.0 else "~NO CHANGE")
        print(f"  DELTA: {delta:+.2f} pts -- {verdict}")