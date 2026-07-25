import numpy as np


# ============================================================
# 1. RISK CLASSIFICATION ENGINE (Dynamic)
# ============================================================
class RiskClassificationModel:
    """
    Binary Classification: 1 (High Risk) or 0 (Safe).
    Determines risk based on structural complexity, voltage drop rates,
    and recent failure density rather than static mock values.
    """

    def __init__(self, critical_voltage=2.4, max_complexity=3.0):
        self.critical_voltage = critical_voltage
        self.max_complexity = max_complexity

    def predict(self, features: dict) -> int:
        """
        features expected:
        - current_voltage (float)
        - voltage_drop_rate (float)
        - structural_complexity (int)
        - is_loop_header (bool)
        """
        voltage = features.get("current_voltage", 3.0)
        drop_rate = features.get("voltage_drop_rate", 0.0)
        complexity = features.get("structural_complexity", 1)
        is_loop = features.get("is_loop_header", False)

        # Immediate hardware risk
        if voltage < self.critical_voltage:
            return 1

        # Predictive hardware risk (steep voltage drop)
        if drop_rate > 0.15:  # e.g., dropping 150mV per cycle block
            return 1

        # Structural risk: Highly complex blocks or tight loops are dangerous to recompute
        if complexity >= self.max_complexity or is_loop:
            return 1

        return 0


# ============================================================
# 2. EFFICIENCY REGRESSION ENGINE (Stateful EMA)
# ============================================================
class EfficiencyRegressionModel:
    """
    Continuous Regression: 0.0 to 1.0.
    Predicts the operational efficiency if a checkpoint is taken NOW.
    Uses an Exponential Moving Average (EMA) tuned via the 'alpha' parameter.
    """

    def __init__(self, alpha=0.45):
        # Alpha is the historical weight (tuned via Grid Search)
        self.alpha = alpha

        # Stateful trackers
        self.ema_efficiency = 1.0  # Start fully optimistic
        self.historical_work_done = 0

    def predict(self, features: dict) -> float:
        """
        features expected:
        - work_since_last_cp (int/float)
        - overhead_cost (int/float)
        """
        work_done = features.get("work_since_last_cp", 1.0)
        cp_cost = features.get("overhead_cost", 0.01)

        # Prevent divide-by-zero on absolute first cycle
        if work_done <= 0:
            return 0.0

        # Instantaneous Efficiency = Useful Work / (Useful Work + Overhead)
        instant_efficiency = work_done / (work_done + cp_cost)

        # Update Stateful EMA
        # EMA_new = (alpha * current_observation) + ((1 - alpha) * EMA_old)
        self.ema_efficiency = (self.alpha * instant_efficiency) + ((1 - self.alpha) * self.ema_efficiency)

        # Add a slight penalty if the checkpoint cost represents more than 20% of the work done
        # This stops the model from thrashing (checkpointing every single cycle)
        if cp_cost > (0.20 * work_done):
            penalty = 0.10
        else:
            penalty = 0.0

        predicted = self.ema_efficiency - penalty

        return max(0.0, min(predicted, 1.0))


# ============================================================
# 3. HYBRID DECISION CONTROLLER
# ============================================================
class HybridCheckpointModel:
    """
    Production Hybrid Model:
    Combines Risk Classification (Binary) with Stateful Efficiency Regression (EMA).
    Accepts Alpha and Threshold parameters directly from the tuning pipeline.
    """

    def __init__(self, alpha=0.45, threshold=0.35):
        self.alpha = alpha
        self.threshold = threshold

        # Instantiate real engines instead of mocks
        self.clf_model = RiskClassificationModel()
        self.reg_model = EfficiencyRegressionModel(alpha=self.alpha)

    def should_checkpoint(self, features: dict) -> bool:
        """
        Master decision gate.
        Returns True ONLY if structural/hardware risk is High AND
        predicted mathematical efficiency remains above the tuned threshold.
        """
        # 1. Classification Engine (Risk)
        is_risky = self.clf_model.predict(features)

        # 2. Regression Engine (Efficiency)
        predicted_efficiency = self.reg_model.predict(features)

        # 3. Dual-Gate Logic
        if is_risky == 1 and predicted_efficiency >= self.threshold:
            return True

        return False
