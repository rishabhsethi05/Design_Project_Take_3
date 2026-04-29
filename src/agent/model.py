import numpy as np
import random
import pickle


class AdaptiveCheckpointAgent:
    """
    Ported from Design Project Take 2.
    Enhanced with Physical Energy Awareness for Mapi-Pro.
    """

    def __init__(self, actions=[0, 1], learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.actions = actions  # 0: Execute, 1: Checkpoint
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def _get_state_key(self, voltage, pc_percent, inflow_mw):
        """
        Discretization logic ported from previous project.
        Voltage replaces 'Stochastic Risk'.
        """
        # We use finer bins for voltage (0.05V) to capture the discharge curve
        v_bin = round(voltage * 20) / 20
        pc_bin = round(pc_percent / 10) * 10
        # Inflow status: 0 (No sun), 1 (Weak/Winter), 2 (Strong/Summer)
        inflow_bin = 2 if inflow_mw > 0.5 else (1 if inflow_mw > 0.05 else 0)

        return (v_bin, pc_bin, inflow_bin)

    def choose_action(self, voltage, pc_percent, inflow_mw):
        state = self._get_state_key(voltage, pc_percent, inflow_mw)

        if state not in self.q_table:
            # Initialize with small optimistic values to encourage exploration
            self.q_table[state] = np.zeros(len(self.actions))

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, voltage, pc_percent, inflow_mw, action, status, cycles_spent,
              next_voltage, next_pc, next_inflow):
        """
        The Brain: Bellman Equation using Time-Minimization Reward.
        Matches the 'Efficiency' metric from Project Take 2.
        """
        state = self._get_state_key(voltage, pc_percent, inflow_mw)
        next_state = self._get_state_key(next_voltage, next_pc, next_inflow)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))

        # --- REWARD CALCULATION ---
        # We minimize Total Execution Time: T = T_compute + T_overhead + T_recovery
        if status == "CRASH":
            # RECOVERY COST: Massive penalty for losing all cycles since last CP
            reward = -20000
        elif action == 1:
            # OVERHEAD COST: Cost of writing to FRAM
            reward = -cycles_spent
        else:
            # COMPUTATION COST: The goal is to accumulate as few negative points as possible
            reward = -cycles_spent

            # Bellman Update
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])

        # New Q-Value
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

    def save_model(self, filename="mapi_pro_agent.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)