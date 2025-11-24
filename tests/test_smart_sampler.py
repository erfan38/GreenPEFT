# test_smart_sampler.py
"""
Unit tests for Smart Sampler classes
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch
import sys
import os

# Setup path for importing  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energypeft.core.smart_sampler import EnergyAwareSampler, GradientImportanceTracker


class TestGradientImportanceTracker(unittest.TestCase):
    """Test cases for GradientImportanceTracker"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dataset_size = 1000
        self.tracker = GradientImportanceTracker(self.dataset_size)
    
    def test_initialization(self):
        """Test tracker initialization"""
        self.assertEqual(self.tracker.dataset_size, self.dataset_size)
        self.assertEqual(len(self.tracker.importance_scores), self.dataset_size)
        self.assertTrue(np.all(self.tracker.importance_scores == 1.0))  # Initial scores should be 1.0
    
    def test_update_importance(self):
        """Test importance score updates"""
        sample_indices = [0, 1, 2]
        gradients = torch.tensor([0.5, 1.0, 2.0])
        
        initial_scores = self.tracker.importance_scores[sample_indices].copy()
        self.tracker.update_importance(sample_indices, gradients)
        updated_scores = self.tracker.importance_scores[sample_indices]
        
        # Scores should have changed
        self.assertFalse(np.array_equal(initial_scores, updated_scores))
        
        # Higher gradient should result in higher importance
        self.assertGreater(updated_scores[2], updated_scores[0])
    
    def test_get_normalized_scores(self):
        """Test normalized importance scores"""
        # Update some scores to create variation
        sample_indices = [0, 1, 2]
        gradients = torch.tensor([0.1, 0.5, 1.0])
        self.tracker.update_importance(sample_indices, gradients)
        
        normalized = self.tracker.get_normalized_scores()
        
        # Should be in range [0, 1]
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))


class TestEnergyAwareSampler(unittest.TestCase):
    """Test cases for EnergyAwareSampler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dataset_size = 1000
        self.base_batch_size = 32
        
        # Create mock energy monitor
        self.mock_energy_monitor = Mock()
        self.mock_energy_monitor.get_remaining_energy.return_value = 50.0
        
        self.sampler = EnergyAwareSampler(
            dataset_size=self.dataset_size,
            energy_monitor=self.mock_energy_monitor,
            base_batch_size=self.base_batch_size,
            importance_weight=0.7
        )
    
    def test_initialization(self):
        """Test sampler initialization"""
        self.assertEqual(self.sampler.dataset_size, self.dataset_size)
        self.assertEqual(self.sampler.base_batch_size, self.base_batch_size)
        self.assertEqual(self.sampler.importance_weight, 0.7)
        self.assertEqual(len(self.sampler.available_indices), self.dataset_size)
    
    def test_sample_batch_basic(self):
        """Test basic batch sampling"""
        batch = self.sampler.sample_batch(target_batch_size=16)
        
        # Should return requested batch size or smaller
        self.assertLessEqual(len(batch), 16)
        self.assertGreater(len(batch), 0)
        
        # All indices should be valid
        self.assertTrue(all(0 <= idx < self.dataset_size for idx in batch))
        
        # Should be unique indices
        self.assertEqual(len(batch), len(set(batch)))
    
    def test_energy_constraints(self):
        """Test energy-constrained sampling"""
        # Set very low remaining energy
        self.mock_energy_monitor.get_remaining_energy.return_value = 0.001
        
        batch = self.sampler.sample_batch(target_batch_size=32)
        
        # Should return very small batch or empty due to energy constraints
        self.assertLessEqual(len(batch), 5)
    
    def test_without_replacement_sampling(self):
        """Test without-replacement sampling logic"""
        all_sampled = set()
        
        # Sample multiple batches
        for _ in range(5):
            batch = self.sampler.sample_batch(target_batch_size=50)
            
            # Check no overlap with previously sampled in same epoch
            overlap = all_sampled.intersection(set(batch))
            self.assertEqual(len(overlap), 0, "Should not sample same indices in same epoch")
            
            all_sampled.update(batch)
        
        # If we've sampled close to full dataset, should reset
        if len(all_sampled) > self.dataset_size * 0.8:
            # Next batch should potentially include previously sampled indices
            batch = self.sampler.sample_batch(target_batch_size=50)
            self.assertGreater(len(batch), 0)
    
    def test_importance_sampling(self):
        """Test importance-based sampling"""
        # Create mock importance scores with clear differences
        importance_scores = np.random.rand(self.dataset_size)
        importance_scores[0:10] = 0.9  # High importance samples
        importance_scores[10:] = 0.1   # Low importance samples
        
        # Sample multiple times and check if high-importance samples are favored
        high_importance_count = 0
        total_samples = 0
        
        for _ in range(10):
            batch = self.sampler.sample_batch(
                target_batch_size=20,
                importance_scores=importance_scores
            )
            
            # Count high importance samples (indices 0-9)
            high_importance_in_batch = sum(1 for idx in batch if idx < 10)
            high_importance_count += high_importance_in_batch
            total_samples += len(batch)
        
        # High importance samples should be overrepresented
        high_importance_ratio = high_importance_count / total_samples
        expected_random_ratio = 10 / self.dataset_size  # 10 high-importance out of 1000
        
        # Should sample high-importance more than random
        self.assertGreater(high_importance_ratio, expected_random_ratio * 2)
    
    def test_update_gradient_importance(self):
        """Test gradient importance updates"""
        sample_indices = [0, 1, 2]
        gradients = torch.tensor([0.5, 1.0, 1.5])
        
        initial_scores = self.sampler.get_importance_scores().copy()
        self.sampler.update_gradient_importance(sample_indices, gradients)
        updated_scores = self.sampler.get_importance_scores()
        
        # Scores should have been updated
        self.assertFalse(np.array_equal(initial_scores, updated_scores))


class TestSamplerIntegration(unittest.TestCase):
    """Integration tests for sampler components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.mock_energy_monitor = Mock()
        self.mock_energy_monitor.get_remaining_energy.return_value = 100.0
        
        self.sampler = EnergyAwareSampler(
            dataset_size=100,
            energy_monitor=self.mock_energy_monitor,
            base_batch_size=10
        )
    
    def test_full_sampling_cycle(self):
        """Test complete sampling cycle with importance updates"""
        # Initial sampling
        batch1 = self.sampler.sample_batch(target_batch_size=10)
        self.assertEqual(len(batch1), 10)
        
        # Simulate gradient update
        gradients = torch.randn(len(batch1))
        self.sampler.update_gradient_importance(batch1, gradients)
        
        # Sample again with updated importance
        importance_scores = self.sampler.get_importance_scores()
        batch2 = self.sampler.sample_batch(
            target_batch_size=10,
            importance_scores=importance_scores
        )
        
        self.assertEqual(len(batch2), 10)
        # Should be different samples (without replacement)
        self.assertEqual(len(set(batch1).intersection(set(batch2))), 0)


if __name__ == '__main__':
    unittest.main()