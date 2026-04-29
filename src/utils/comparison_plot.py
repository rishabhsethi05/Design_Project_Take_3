import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_research_comparison():
    scenarios = ["summer", "winter", "night"]
    plt.figure(figsize=(12, 6))

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    for sc in scenarios:
        file_path = os.path.join(base_path, f"output_trace_{sc}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Group by epoch to get the best performance per training round
            epoch_max = df.groupby('Epoch')['Total_Cycles'].max()
            plt.plot(epoch_max.index, epoch_max.values, label=f'Scenario: {sc.capitalize()}', marker='o')

    plt.title("Mapi-Pro ML Agent: Execution Efficiency Across Scenarios", fontsize=14)
    plt.xlabel("Training Epochs", fontsize=12)
    plt.ylabel("Total Clock Cycles (Latency)", fontsize=12)
    plt.yscale('log')  # Log scale often shows learning curves better
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    save_path = os.path.join(base_path, "multi_scenario_comparison.png")
    plt.savefig(save_path)
    print(f"[*] Comparison plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    generate_research_comparison()