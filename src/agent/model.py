import numpy as np
import pickle

class AdaptiveCheckpointAgent:
    """
    MAPI-PRO Adaptive Agent
    Optimized for maximum Efficiency Gain by balancing aggressive execution
    against strategic checkpoint placement.
    """
    def __init__(self, actions=[0, 1], learning_rate=0.1, discount_factor=0.9, epsilon=0.05):
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def _get_state_key(self, voltage, pc_percent):
        """
        Simplifies the physical environment into discrete states.
        Focuses on Voltage and Progress to maximize convergence speed.
        """
        v_bin = round(voltage, 1)
        # Using 20% bins reduces the Q-table size, helping the agent learn faster
        pc_bin = int(pc_percent / 20) * 20
        return (v_bin, pc_bin)

    def choose_action(self, voltage, pc_percent, inflow, complexity=1):
        """
        Selects next action: 0 (Execute) or 1 (Checkpoint).
        Complexity is accepted to maintain compatibility with sim_engine.
        """
        state = self._get_state_key(voltage, pc_percent)

        # Epsilon-greedy exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)

        # Retrieve Q-values, defaulting to high-reward for execution (Action 0)
        return np.argmax(self.q_table.get(state, [2.0, 0.0]))

    def learn(self, v, pc, inflow, action, status, cost, v_next, pc_next, inflow_next, complexity=1):
        """
        Updates the Q-table based on the transition outcome.
        Reward structure is tuned to favor execution and penalize progress loss (Crashes).
        """
        state = self._get_state_key(v, pc)
        next_state = self._get_state_key(v_next, pc_next)

        # REWARD MAPPING
        if status == "CRASH":
            reward = -10000  # Doubled penalty (was -5000)
        elif action == 1:
            reward = -50  # Drastically lower CP cost (was -200)
        else:
            reward = 500   # Reward for successful execution steps

        # Initialize state in Q-table if new
        if state not in self.q_table:
            self.q_table[state] = [2.0, 0.0]

        # Q-Learning Update (Bellman Equation)
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table.get(next_state, [2.0, 0.0]))

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

    def save(self, filename="mapi_pro_agent.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def save_model(self, filename="mapi_pro_agent.pkl"):
        self.save(filename)