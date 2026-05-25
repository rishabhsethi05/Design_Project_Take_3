import matplotlib.pyplot as plt
import numpy as np


scenarios = ['Summer', 'Winter', 'Night']

baseline = [33867782, 64267630, 0]
optimized = [16000527, 16000240, 0]

x = np.arange(len(scenarios))

width = 0.35

fig, ax = plt.subplots(figsize=(10,6))

bars1 = ax.bar(x - width/2, baseline, width,
               label='Baseline')

bars2 = ax.bar(x + width/2, optimized, width,
               label='Our Model')

ax.set_title(
    'Baseline vs Our Model Execution Cost',
    fontsize=16,
    fontweight='bold'
)

ax.set_xlabel('Scenario', fontsize=13)
ax.set_ylabel('Clock Cycles', fontsize=13)

ax.set_xticks(x)
ax.set_xticklabels(scenarios)

ax.legend()

ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

plt.savefig("final_performance_comparison.png", dpi=600)

plt.show()