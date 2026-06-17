import numpy as np
import pickle


class AdaptiveCheckpointAgent:
    """
    MAPI-PRO Adaptive Agent
    Optimized for maximum Efficiency Gain by balancing aggressive execution
    against strategic checkpoint placement using context-aware energy rewards.
    """

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, reward_config=None):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.actions = [0, 1]
        # Default to our current baseline if no config is provided
        if reward_config is None:
            self.reward_config = {
                'crash': -12890.2,
                'summer_cp': -792.2,
                'winter_cp': -57.6,
                'execute': 9.2
            }
        else:
            self.reward_config = reward_config

    def _get_state_key(self, voltage, pc_percent, inflow, complexity):
        """
        Simplifies the physical environment into discrete states.
        Incorporates an energy harvesting inflow indicator to allow the agent
        to distinguish between abundant (Summer) and scarce (Winter) conditions.
        """
        # 1. Voltage: Keep 1 decimal place (e.g., 2.5, 2.6)
        v_bin = round(voltage, 1)

        # 2. RESOLUTION FIX: Shrink bins from 20% to 4%
        # This reduces the agent's blind spot from ~57 instructions to ~11
        pc_bin = int(pc_percent / 4) * 4

        # 3. COMPLEXITY FIX: Add code structure to the Q-table!
        # Cap at 3 to prevent the Q-table from exploding in size
        comp_bin = min(int(complexity), 3)

        # 4. Contextual Energy Flag
        inflow_bin = 1 if inflow >= 0.1 else 0

        return (v_bin, pc_bin, inflow_bin, comp_bin)

    def choose_action(self, voltage, pc_percent, inflow, complexity=1):
        """
        Selects next action: 0 (Execute) or 1 (Checkpoint).
        """
        # Pass complexity into the state key
        state = self._get_state_key(voltage, pc_percent, inflow, complexity)

        # Epsilon-greedy exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)

        # Retrieve Q-values, defaulting to high-reward for execution (Action 0)
        return np.argmax(self.q_table.get(state, [2.0, 0.0]))

    def learn(self, v, pc, inflow, action, status, cost, v_next, pc_next, inflow_next, complexity=1):
        state = self._get_state_key(v, pc, inflow, complexity)
        next_state = self._get_state_key(v_next, pc_next, inflow_next, complexity)
        is_summer = state[2] == 1

        if status == "CRASH":
            reward = self.reward_config["crash"]
        elif action == 1: # CHECKPOINT
            reward = self.reward_config["summer_cp"] if is_summer else self.reward_config["winter_cp"]
        else: # EXECUTE
            reward = self.reward_config["execute"]

        if state not in self.q_table:
            self.q_table[state] = [2.0, 0.0]

        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table.get(next_state, [2.0, 0.0]))
        self.q_table[state][action] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)


    def save(self, filename="mapi_pro_agent.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def save_model(self, filename="mapi_pro_agent.pkl"):
        self.save(filename)