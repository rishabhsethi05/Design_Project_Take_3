import numpy as np


class EnergyHarvester:
    """
    Simulates energy harvesting profiles for Intermittent Computing.
    Supports three scenarios: Summer (Stable), Winter (Stochastic), Night (Zero).
    """

    def __init__(self, scenario="Summer"):
        self.scenario = scenario

        # Power constants in milliwatts (mW)
        # Based on typical small solar cells for MSP430-scale IoT
        self.P_HIGH = 2.0  # Clear sun
        self.P_LOW = 0.2  # Overcast/Shadow
        self.P_ZERO = 0.0  # Night

        # Stochastic state for Winter/Rainy scenario
        self.current_state_power = self.P_HIGH
        self.transition_prob = 0.15  # Probability of weather change per step

    def set_scenario(self, scenario):
        if scenario in ["Summer", "Winter", "Night"]:
            self.scenario = scenario
        else:
            raise ValueError("Scenario must be Summer, Winter, or Night")

    def get_inflow_power(self):
        """
        Returns current power inflow in milliwatts (mW) based on scenario.
        """
        if self.scenario == "Summer":
            # Stable solar: Constant high power
            return self.P_HIGH

        elif self.scenario == "Night":
            # Zero energy available
            return self.P_ZERO

        elif self.scenario == "Winter":
            # Stochastic: Markovian transition between High and Low power
            if np.random.rand() < self.transition_prob:
                self.current_state_power = self.P_LOW if self.current_state_power == self.P_HIGH else self.P_HIGH
            return self.current_state_power

    def step_harvest(self, capacitor, duration_ms):
        """
        Calculates energy harvested over duration_ms and updates the capacitor.
        :param capacitor: Instance of MSP430Capacitor
        :param duration_ms: Time step in milliseconds
        """
        power_mw = self.get_inflow_power()
        capacitor.harvest_energy(power_mw, duration_ms)
        return power_mw