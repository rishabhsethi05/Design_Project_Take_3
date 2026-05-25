import random
import numpy as np


class MockRegressionModel:
    """
    Simulates efficiency prediction.
    Returns a score between 0 and 1.
    """

    def predict(self, features):
        voltage = features["voltage"]
        progress = features["progress"]
        complexity = features["complexity"]

        # Higher voltage + higher complexity -> higher urgency
        score = (
            0.45 * voltage / 3.3 +
            0.35 * complexity / 5.0 +
            0.20 * progress
        )

        return min(score, 1.0)


class MockClassificationModel:
    """
    Simulates risk prediction.
    Returns:
    1 -> risky
    0 -> safe
    """

    def predict(self, features):
        voltage = features["voltage"]
        complexity = features["complexity"]

        if voltage < 2.4:
            return 1

        if complexity >= 3:
            return 1

        return 0


class HybridCheckpointModel:
    """
    Hybrid Baseline Model:
    Classification + Regression checkpointing.
    """

    def __init__(self, threshold=0.82):
        self.reg_model = MockRegressionModel()
        self.clf_model = MockClassificationModel()
        self.threshold = threshold

    def should_checkpoint(self, features):

        is_risky = self.clf_model.predict(features)

        predicted_efficiency = self.reg_model.predict(features)

        if is_risky == 1 and predicted_efficiency >= self.threshold:
            return True

        return False