import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def smooth_curve(data, window=5):
    return data.rolling(window=window, min_periods=1).mean()


def generate_research_comparison():

    scenarios = ["summer", "winter"]

    plt.figure(figsize=(11, 6))

    base_path = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )

    for sc in scenarios:

        file_path = os.path.join(base_path, f"output_trace_{sc}.csv")

        if os.path.exists(file_path):

            df = pd.read_csv(file_path)

            epoch_cycles = df.groupby('Epoch')['Total_Cycles'].max()

            smoothed = smooth_curve(epoch_cycles)

            plt.plot(
                epoch_cycles.index,
                smoothed,
                linewidth=2.5,
                marker='o',
                markersize=3,
                label=f'{sc.capitalize()} Scenario'
            )

    plt.title(
        "MAPI-PRO Reinforcement Learning Convergence",
        fontsize=16,
        fontweight='bold'
    )

    plt.xlabel(
        "Training Epoch",
        fontsize=13
    )

    plt.ylabel(
        "Total Execution Cycles",
        fontsize=13
    )

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend(fontsize=11)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()

    save_path = os.path.join(base_path, "rl_convergence_plot.png")

    plt.savefig(save_path, dpi=600)

    print(f"Saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    generate_research_comparison()