import torch
import numpy as np
from torch.utils.data import Sampler
from typing import List, Optional, Iterator

class LossEfficiencyTracker:
    """
    Tracks Loss (Utility) and Length (Cost) to compute Value-per-Watt scores.
    Replaces the expensive GradientImportanceTracker.
    """
    
    def __init__(self, dataset_size: int, alpha: float = 1.0, beta: float = 0.5, decay: float = 0.9):
        self.dataset_size = dataset_size
        self.alpha = alpha # Importance of high loss (Learning Opportunity)
        self.beta = beta   # Penalty for sequence length (Energy Cost)
        self.decay = decay # Momentum for moving average
        
        # Initialize scores to 1.0 so all samples have equal chance initially
        self.efficiency_scores = np.ones(dataset_size, dtype=np.float32)
        
    def update_batch_outcomes(self, indices: List[int], losses: List[float], lengths: List[int]):
        """
        Update scores based on the actual training result.
        Equation: Score = (Loss^alpha) / (Length^beta)
        """
        for idx, loss, length in zip(indices, losses, lengths):
            if 0 <= idx < self.dataset_size:
                # Avoid division by zero or negative lengths
                safe_len = max(1, length)
                
                # Calculate Value-per-Watt
                # High Loss = Good (We need to learn this)
                # High Length = Bad (It costs a lot of energy)
                new_score = (loss ** self.alpha) / (safe_len ** self.beta)
                
                # Update with exponential moving average to keep history stable
                self.efficiency_scores[idx] = (
                    self.decay * self.efficiency_scores[idx] + 
                    (1 - self.decay) * new_score
                )
    
    def get_probabilities(self) -> np.ndarray:
        """Get normalized sampling probabilities"""
        scores = self.efficiency_scores
        total_score = scores.sum()
        
        # Normalize to create a valid probability distribution
        if total_score > 0:
            return scores / total_score
        
        # Fallback to uniform if scores are invalid
        return np.ones_like(scores) / len(scores)

class EnergyAwareSampler(Sampler):
    """
    Custom PyTorch Sampler that selects data based on 'Value-per-Watt'.
    Inherits from Sampler for compatibility with DataLoader.
    """
    def __init__(self, dataset, energy_monitor, base_batch_size=32):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.energy_monitor = energy_monitor
        self.base_batch_size = base_batch_size
        
        # The "Brain" that tracks which samples are efficient
        self.tracker = LossEfficiencyTracker(self.dataset_size)
        
        # Internal state
        self.epoch_indices = list(range(self.dataset_size))
        self.used_indices = set()

    def update_batch_outcomes(self, indices: List[int], losses: List[float], lengths: List[int]):
        """
        Public method to receive feedback from the Trainer.
        """
        self.tracker.update_batch_outcomes(indices, losses, lengths)

    def __iter__(self) -> Iterator[int]:
        """
        Standard PyTorch Sampler iterator.
        1. Calculate Probabilities based on historical efficiency.
        2. Sample without replacement.
        """
        # 1. Get probabilities from our Tracker
        # These are based on the LAST time we saw these samples
        probs = self.tracker.get_probabilities()
        
        # 2. Reset tracking for the new epoch
        self.used_indices.clear()
        remaining_indices = list(range(self.dataset_size))
        
        # 3. Loop until we have used all data
        while len(remaining_indices) > 0:
            # Determine batch size (placeholder for adaptive logic if needed)
            batch_size = self.base_batch_size 
            
            # We need to re-normalize probabilities for ONLY the remaining indices
            # (Because we are sampling without replacement)
            current_probs = probs[remaining_indices]
            prob_sum = current_probs.sum()
            
            if prob_sum > 0:
                current_probs = current_probs / prob_sum
            else:
                current_probs = None # Fallback to uniform
            
            # Select indices for this batch
            # We select 'batch_size' items from 'remaining_indices'
            # Note: np.random.choice handles p=None as uniform
            selected_indices_local = np.random.choice(
                len(remaining_indices), 
                size=min(len(remaining_indices), batch_size), 
                replace=False, 
                p=current_probs
            )
            
            # Map back to global dataset indices
            selected_indices_global = [remaining_indices[i] for i in selected_indices_local]
            
            # Yield indices for the DataLoader to fetch
            yield from selected_indices_global
            
            # Remove used indices from the pool (Without Replacement)
            # Rebuilding list is robust and cleaner for Python lists
            remaining_set = set(remaining_indices) - set(selected_indices_global)
            remaining_indices = list(remaining_set)

    def __len__(self) -> int:
        """Required by DataLoader to calculate steps per epoch"""
        return self.dataset_size