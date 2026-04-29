import pandas as pd
import matplotlib.pyplot as plt


def plot_simulation_results(csv_file):
    df = pd.read_csv(csv_file)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)

    # --- Plot 1: Voltage Trace for the Final Epoch ---
    # We take the last epoch to see the "learned" behavior
    last_epoch = df['Epoch'].max()
    epoch_data = df[df['Epoch'] == last_epoch].reset_index()

    ax1.plot(epoch_data.index, epoch_data['Voltage'], label='Capacitor Voltage', color='blue', linewidth=1.5)
    ax1.axhline(y=2.0, color='red', linestyle='--', label='V_death (2.0V)')

    # Mark Checkpoints
    checkpoints = epoch_data[epoch_data['Action'] == 1]
    ax1.scatter(checkpoints.index, checkpoints['Voltage'], color='green', marker='d', s=100, label='ML Checkpoint')

    # Mark Crashes
    crashes = epoch_data[epoch_data['Status'] == 'CRASH']
    ax1.scatter(crashes.index, crashes['Voltage'], color='orange', marker='x', s=100, label='System Crash')

    ax1.set_title(f"Voltage Profile & ML Decisions (Epoch {last_epoch})", fontsize=14)
    ax1.set_xlabel("Simulation Steps", fontsize=12)
    ax1.set_ylabel("Voltage (V)", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Learning Efficiency (Total Cycles per Epoch) ---
    # Group by epoch and find the maximum total cycles (cumulative)
    efficiency = df.groupby('Epoch')['Total_Cycles'].max().reset_index()

    ax2.bar(efficiency['Epoch'], efficiency['Total_Cycles'], color='teal', alpha=0.7)
    ax2.set_title("Total Execution Time (Efficiency Metric) over Epochs", fontsize=14)
    ax2.set_xlabel("Epoch (Learning Phase)", fontsize=12)
    ax2.set_ylabel("Total Clock Cycles", fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    # Annotate the improvement
    first_val = efficiency['Total_Cycles'].iloc[0]
    last_val = efficiency['Total_Cycles'].iloc[-1]
    improvement = ((first_val - last_val) / first_val) * 100
    ax2.text(0.5, 0.9, f"Improvement: {improvement:.2f}%", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', color='darkgreen')

    plt.savefig("research_results.png")
    print("Graphs saved as 'research_results.png'.")
    plt.show()


if __name__ == "__main__":
    # NEW CODE
    import os

    # Get the path to the root directory (two levels up from this script)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path = os.path.join(root_dir, "output_trace_winter.csv")

    plot_simulation_results(csv_path)