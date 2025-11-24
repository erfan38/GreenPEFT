# early_stopper.py
import numpy as np

class EnergyEfficiencyEarlyStopping:
    """
    Stops training if improvement per unit energy drops below a threshold for a set number of checks (patience).
    """
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience  # Number of checks to wait before stopping
        self.min_delta = min_delta  # Minimum improvement per Wh to count as progress
        self.counter = 0
        self.best_efficiency = -np.inf
        self.early_stop = False

    def __call__(self, loss_history, energy_history):
        # Only check if we have enough history
        if len(loss_history) < 2 or len(energy_history) < 2:
            return False
        # Calculate recent improvement and energy used
        recent_improvement = loss_history[-2] - loss_history[-1]
        recent_energy = energy_history[-1] - energy_history[-2]
        if recent_energy <= 0:
            return False
        efficiency = recent_improvement / recent_energy
        # Check if efficiency improved enough
        if efficiency > self.best_efficiency + self.min_delta:
            self.best_efficiency = efficiency
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
