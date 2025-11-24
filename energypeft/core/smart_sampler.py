import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque

class GradientImportanceTracker:
    """Track gradient importance for smart sampling"""
    
    def __init__(self, dataset_size: int, decay_factor: float = 0.95):
        self.dataset_size = dataset_size
        self.decay_factor = decay_factor
        
        # Initialize importance scores
        self.importance_scores = np.ones(dataset_size, dtype=np.float32)
        self.last_seen = np.zeros(dataset_size, dtype=np.int32)
        self.update_count = 0
        
    def update_importance(self, sample_indices: List[int], gradients: torch.Tensor):
        """Update importance scores based on gradient magnitudes"""
        self.update_count += 1
        
        # Calculate gradient norms for each sample
        if gradients.dim() > 1:
            grad_norms = torch.norm(gradients.view(gradients.size(0), -1), dim=1).cpu().numpy()
        else:
            grad_norms = torch.abs(gradients).cpu().numpy()
            
        for idx, grad_norm in zip(sample_indices, grad_norms):
            if 0 <= idx < self.dataset_size:
                # Decay old importance
                staleness = self.update_count - self.last_seen[idx]
                decay = self.decay_factor ** staleness
                
                # Update with exponential moving average
                self.importance_scores[idx] = (
                    decay * self.importance_scores[idx] + 
                    (1 - decay) * grad_norm
                )
                self.last_seen[idx] = self.update_count
    
    def get_importance_scores(self) -> np.ndarray:
        """Get current importance scores"""
        return self.importance_scores.copy()
    
    def get_normalized_scores(self) -> np.ndarray:
        """Get importance scores normalized to [0, 1]"""
        scores = self.importance_scores
        if scores.max() > scores.min():
            return (scores - scores.min()) / (scores.max() - scores.min())
        return scores

class EnergyAwareSampler:
    """Energy-constrained gradient importance sampling"""
    
    def __init__(self, 
                 dataset_size: int, 
                 energy_monitor,
                 base_batch_size: int = 32,
                 importance_weight: float = 0.7):
        
        self.dataset_size = dataset_size
        self.energy_monitor = energy_monitor
        self.base_batch_size = base_batch_size
        self.importance_weight = importance_weight
        
        # Gradient importance tracking
        self.importance_tracker = GradientImportanceTracker(dataset_size)
        
        # Without-replacement sampling state
        self.available_indices = list(range(dataset_size))
        self.current_epoch_used = set()
        
        # Energy calibration
        self.energy_per_sample = self._calibrate_energy_per_sample()
        
    def _calibrate_energy_per_sample(self) -> float:
        """Calibrate energy consumption per sample (simplified)"""
        # This would be calibrated during first few batches
        # For now, use a reasonable default
        return 0.01  # 0.01 Wh per sample (will be updated dynamically)
    
    def sample_batch(self, 
                    target_batch_size: Optional[int] = None,
                    importance_scores: Optional[np.ndarray] = None) -> List[int]:
        """Sample batch with energy constraints and importance weighting"""
        
        if target_batch_size is None:
            target_batch_size = self.base_batch_size
            
        # Check energy constraints
        remaining_energy = self.energy_monitor.get_remaining_energy()
        max_affordable_samples = int(remaining_energy / max(self.energy_per_sample, 0.001))
        actual_batch_size = min(target_batch_size, max_affordable_samples)
        
        if actual_batch_size <= 0:
            return []  # No energy left
            
        # Get available indices for without-replacement sampling
        available = [idx for idx in self.available_indices if idx not in self.current_epoch_used]
        
        # Reset epoch if all samples used
        if len(available) < actual_batch_size:
            self.current_epoch_used.clear()
            available = self.available_indices.copy()
            
        # Sample based on importance if available
        if importance_scores is not None and len(importance_scores) == self.dataset_size:
            available_scores = importance_scores[available]
            
            # Combine importance with energy efficiency
            energy_efficiency = self._calculate_energy_efficiency(available)
            combined_scores = (self.importance_weight * available_scores + 
                              (1 - self.importance_weight) * energy_efficiency)
            
            # Convert to probabilities
            probabilities = combined_scores / combined_scores.sum()
            
            # Sample without replacement
            selected_idx = np.random.choice(
                len(available), 
                size=min(actual_batch_size, len(available)),
                replace=False,
                p=probabilities
            )
            selected_samples = [available[i] for i in selected_idx]
        else:
            # Random sampling without replacement
            selected_samples = np.random.choice(
                available, 
                size=min(actual_batch_size, len(available)),
                replace=False
            ).tolist()
        
        # Update used samples
        self.current_epoch_used.update(selected_samples)
        
        return selected_samples
    
    def _calculate_energy_efficiency(self, sample_indices: List[int]) -> np.ndarray:
        """Calculate energy efficiency scores for samples"""
        # Simplified: assume uniform energy efficiency
        # In practice, this could consider model complexity, sample difficulty, etc.
        return np.ones(len(sample_indices), dtype=np.float32)
    
    def update_gradient_importance(self, sample_indices: List[int], gradients: torch.Tensor):
        """Update importance scores after backward pass"""
        self.importance_tracker.update_importance(sample_indices, gradients)
        
    def get_importance_scores(self) -> np.ndarray:
        """Get current importance scores"""
        return self.importance_tracker.get_normalized_scores()
