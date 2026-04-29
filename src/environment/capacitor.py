import math


class MSP430Capacitor:
    """
    Simulates the physical 100uF capacitor for the TI MSP430FR6989.
    Tracks Energy (Joules) and Voltage (V) based on: E = 0.5 * C * V^2
    """

    def __init__(self, capacity_uf=100.0, v_max=3.3, v_min=2.0):
        # Hardware Constants
        self.C = capacity_uf * 1e-6  # Convert to Farads
        self.V_MAX = v_max  # Upper rail voltage
        self.V_DEATH = v_min  # System death threshold (Hard Stop)

        # Initial State
        self.v_curr = v_max
        self.is_dead = False
        self.total_energy_consumed_nj = 0.0

    def get_energy(self, voltage):
        """Calculates energy in Joules for a given voltage."""
        return 0.5 * self.C * (voltage ** 2)

    def get_current_voltage(self):
        """Returns the current voltage level."""
        return self.v_curr

    def consume_energy(self, energy_nj):
        """
        Drains energy from the capacitor.
        :param energy_nj: Energy consumed in nanojoules.
        """
        if self.is_dead:
            return

        self.total_energy_consumed_nj += energy_nj
        energy_j = energy_nj * 1e-9

        current_energy = self.get_energy(self.v_curr)
        new_energy = current_energy - energy_j

        # Check for Hard Stop / Brownout
        if self.v_curr <= self.V_DEATH or new_energy <= self.get_energy(self.V_DEATH):
            self.v_curr = self.V_DEATH
            self.is_dead = True
        else:
            # V = sqrt(2E / C)
            self.v_curr = math.sqrt((2 * new_energy) / self.C)

    def harvest_energy(self, power_mw, time_ms):
        """
        Adds energy to the capacitor via harvesting.
        :param power_mw: Harvesting power in milliwatts.
        :param time_ms: Time duration of harvesting in milliseconds.
        """
        # Energy (J) = Power (W) * Time (s)
        harvested_j = (power_mw * 1e-3) * (time_ms * 1e-3)

        current_energy = self.get_energy(self.v_curr)
        new_energy = current_energy + harvested_j

        # Cap at V_MAX
        max_energy = self.get_energy(self.V_MAX)
        if new_energy > max_energy:
            new_energy = max_energy

        self.v_curr = math.sqrt((2 * new_energy) / self.C)

        # If we were dead and voltage rose above V_DEATH, we are alive again
        if self.v_curr > self.V_DEATH:
            self.is_dead = False

    def reset_to_full(self):
        """Resets the capacitor for a new simulation run."""
        self.v_curr = self.V_MAX
        self.is_dead = False
        self.total_energy_consumed_nj = 0.0

    def __repr__(self):
        status = "ALIVE" if not self.is_dead else "DEAD"
        return f"Capacitor(V: {self.v_curr:.3f}V, State: {status})"