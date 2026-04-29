class RewardSystem:
    """
    Calculates rewards based on the 'Minimizing Total Execution Time' objective.
    """

    def __init__(self):
        # Penalty constants (Negative rewards)
        self.PENALTY_CRASH = -5000.0  # Massive penalty for losing progress
        self.PENALTY_STEP = -1.0  # Small penalty for every cycle spent
        self.PENALTY_BACKUP = -50.0  # Penalty for backup overhead
        self.REWARD_COMPLETE = 1000.0  # Big bonus for completing the task

    def calculate(self, status, action_taken):
        """
        status: "SUCCESS", "CRASH", or "COMPLETED"
        action_taken: 0 (Execute) or 1 (Checkpoint)
        """
        reward = 0

        if status == "CRASH":
            return self.PENALTY_CRASH

        if status == "COMPLETED":
            return self.REWARD_COMPLETE

        # Standard execution step
        reward += self.PENALTY_STEP

        # If we chose to checkpoint, add the overhead penalty
        if action_taken == 1:
            reward += self.PENALTY_BACKUP

        return reward