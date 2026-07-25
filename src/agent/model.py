import numpy as np
import pickle
import os


class AdaptiveCheckpointAgent:
    """
    PACE Core Q-Learning Agent.
    Integrated with hardware reflexes, defensive saving rewards,
    and fast-convergence state binning.
    """

    def __init__(self, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_min=0.02, epsilon_decay=0.9995):
        self.q_table = {}

        # Q-Learning Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Stateful Tracking
        self.steps_since_cp = 0
        self.last_state = None

        # Base Configs for Rewards
        self.safe_v_threshold = 2.4
        self.base_spam_tax = 10.0

    def reset_state(self):
        """Reset operational state tracking between episodes/epochs."""
        self.steps_since_cp = 0
        self.last_state = None

    def decay_epsilon(self):
        """Decay exploration rate smoothly down to the fallback minimum."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _get_state_key(self, voltage, pc_percent, inflow, complexity=1, season_flag=1, features=None):
        """
        Groups similar continuous observations into discrete, manageable states
        so the Q-table can converge cleanly without state-space explosion.
        """
        if voltage < 2.15:
            v_bin = 0
        elif voltage < 2.30:
            v_bin = 1
        elif voltage < 2.50:
            v_bin = 2
        else:
            v_bin = 3

        pc_bin = min(3, int(pc_percent // 30))

        if self.steps_since_cp < 5:
            cp_bin = 0
        elif self.steps_since_cp < 20:
            cp_bin = 1
        else:
            cp_bin = 2

        inflow_bin = 1 if inflow > 0.05 else 0

        return (v_bin, pc_bin, cp_bin, inflow_bin)

    def choose_action(self, v, pc, inflow, complexity=1, season_flag=1, features=None, epsilon=None):
        """
        Selects an action using Epsilon-Greedy logic, overridden by strict hardware/efficiency reflexes.
        """
        if pc < 0.01:
            self.steps_since_cp = 0

        # ==========================================
        # SPRINT & EFFICIENCY REFLEXES
        # ==========================================
        # 1. Never checkpoint if capacitor is nearly full (waste of time/energy).
        # 2. Sprint to the finish line if we are >90% done AND voltage is healthy (> 2.4V).
        # This completely shields short programs from suicidal epsilon-exploration.
        if v > 2.65 or (pc > 90.0 and v > 2.4):
            self.steps_since_cp += 1
            return 0

        # ==========================================
        # CRITICAL HARDWARE REFLEX
        # ==========================================
        # Force a save if voltage drops dangerously low to prevent a crash.
        if v < 2.15:
            self.steps_since_cp = 0
            return 1

        state = self._get_state_key(v, pc, inflow, complexity, season_flag, features)
        self.last_state = state

        if state not in self.q_table:
            self.q_table[state] = [50.0, 50.0]

        current_eps = epsilon if epsilon is not None else self.epsilon

        if np.random.rand() < current_eps:
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(self.q_table[state])

        if action == 1:
            self.steps_since_cp = 0
        else:
            self.steps_since_cp += 1

        return action

    def learn(self, v, pc, inflow, action, status, cost, v_next, pc_next, inflow_next, complexity=1, season_flag=1,
              features=None):
        """
        Updates the Q-table using the Bellman Equation.
        Reward logic perfectly mirrors the Hybrid model's priorities: punish dying rails, reward saves.
        """
        state = self.last_state if self.last_state is not None else self._get_state_key(v, pc, inflow, complexity,
                                                                                        season_flag, features)
        next_state = self._get_state_key(v_next, pc_next, inflow_next, complexity, season_flag, features)

        if state not in self.q_table:
            self.q_table[state] = [50.0, 50.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [50.0, 50.0]

        # ==========================================
        # REBALANCED REWARD FUNCTION
        # ==========================================
        if status == "CRASH" or pc_next < pc:
            reward = -500.0
        else:
            if action == 1:  # ACTION: Checkpoint
                if self.steps_since_cp < 10:
                    reward = -50.0  # Punish back-to-back spam
                elif v < self.safe_v_threshold:
                    reward = 150.0 - (float(cost) * 0.1)  # Reward defensive save near death
                else:
                    reward = -10.0 - (float(self.base_spam_tax) * 0.5)  # Modest tax for saving while completely safe
            else:  # ACTION: Continue
                if v < 2.25:
                    reward = -100.0  # Punish riding the dropping rail without saving
                else:
                    reward = 20.0 - (float(cost) * 0.05)  # Normal progress reward

        # Bellman Update
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        self.q_table[state][action] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

    def save_model(self, filepath="mapi_pro_agent.pkl"):
        """Serialize the Q-table and metadata to disk."""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_model(self, filepath="mapi_pro_agent.pkl"):
        """Load a trained Q-table from disk."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data.get('q_table', {})
                self.epsilon = data.get('epsilon', self.epsilon_min)
                return True
        return False