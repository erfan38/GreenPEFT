# test_adaptive_batcher.py
"""
Unit tests for EnergyAwareBatcher class
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Setup path for importing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energypeft.core.adaptive_batcher import EnergyAwareBatcher


class TestEnergyAwareBatcher(unittest.TestCase):
    """Test cases for EnergyAwareBatcher"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_batch_size = 32
        self.min_batch_size = 1
        
        # Create mock energy monitor
        self.mock_energy_monitor = Mock()
        self.mock_energy_monitor.get_remaining_energy.return_value = 75.0  # 75% energy remaining
        self.mock_energy_monitor.energy_budget_wh = 100.0
        
        self.batcher = EnergyAwareBatcher(
            base_batch_size=self.base_batch_size,
            energy_monitor=self.mock_energy_monitor,
            min_batch_size=self.min_batch_size
        )
    
    def test_initialization(self):
        """Test batcher initialization"""
        self.assertEqual(self.batcher.base_batch_size, self.base_batch_size)
        self.assertEqual(self.batcher.min_batch_size, self.min_batch_size)
        self.assertEqual(self.batcher.training_progress, 0.0)
        self.assertEqual(self.batcher.energy_monitor, self.mock_energy_monitor)
    
    def test_adaptive_batch_size_high_energy(self):
        """Test batch size adaptation with high energy remaining"""
        # High energy remaining (>70%)
        self.mock_energy_monitor.get_remaining_energy.return_value = 80.0
        
        batch_size = self.batcher.get_adaptive_batch_size(convergence_progress=0.2)
        
        # Should increase batch size when energy is high
        self.assertGreaterEqual(batch_size, self.base_batch_size)
        self.assertGreaterEqual(batch_size, self.min_batch_size)
    
    def test_adaptive_batch_size_medium_energy(self):
        """Test batch size adaptation with medium energy remaining"""
        # Medium energy remaining (30-70%)
        self.mock_energy_monitor.get_remaining_energy.return_value = 50.0
        
        batch_size = self.batcher.get_adaptive_batch_size(convergence_progress=0.3)
        
        # Should use standard batch size when energy is medium
        expected_size = int(self.base_batch_size * 1.0 * 1.0)  # energy_factor=1.0, progress_factor=1.0
        self.assertEqual(batch_size, max(self.min_batch_size, expected_size))
    
    def test_adaptive_batch_size_low_energy(self):
        """Test batch size adaptation with low energy remaining"""
        # Low energy remaining (<30%)
        self.mock_energy_monitor.get_remaining_energy.return_value = 20.0
        
        batch_size = self.batcher.get_adaptive_batch_size(convergence_progress=0.1)
        
        # Should decrease batch size when energy is low
        expected_size = int(self.base_batch_size * 0.6)  # energy_factor=0.6
        self.assertEqual(batch_size, max(self.min_batch_size, expected_size))
    
    def test_progress_aware_adaptation(self):
        """Test batch size adaptation based on training progress"""
        self.mock_energy_monitor.get_remaining_energy.return_value = 60.0
        
        # Early training (progress < 0.5)
        early_batch_size = self.batcher.get_adaptive_batch_size(convergence_progress=0.2)
        
        # Late training (progress > 0.8)
        late_batch_size = self.batcher.get_adaptive_batch_size(convergence_progress=0.9)
        
        # Late training should have smaller batches
        self.assertLessEqual(late_batch_size, early_batch_size)
    
    def test_minimum_batch_size_enforcement(self):
        """Test that minimum batch size is always enforced"""
        # Very low energy
        self.mock_energy_monitor.get_remaining_energy.return_value = 1.0
        
        batch_size = self.batcher.get_adaptive_batch_size(convergence_progress=0.9)
        
        # Should never go below minimum
        self.assertGreaterEqual(batch_size, self.min_batch_size)
    
    def test_batch_size_calculation_logic(self):
        """Test the detailed batch size calculation logic"""
        # Test high energy scenario
        self.mock_energy_monitor.get_remaining_energy.return_value = 85.0
        
        batch_size = self.batcher.get_adaptive_batch_size(convergence_progress=0.3)
        
        # Calculate expected size
        energy_factor = 1.2  # High energy
        progress_factor = 1.0  # Early-medium progress
        expected_size = int(self.base_batch_size * energy_factor * progress_factor)
        expected_size = max(self.min_batch_size, expected_size)
        
        self.assertEqual(batch_size, expected_size)
    
    def test_energy_ratio_calculation(self):
        """Test energy ratio calculation"""
        # Test with different energy levels
        test_cases = [
            (90.0, 100.0, 0.9),  # 90% remaining
            (50.0, 100.0, 0.5),  # 50% remaining
            (10.0, 100.0, 0.1),  # 10% remaining
        ]
        
        for remaining_energy, budget, expected_ratio in test_cases:
            self.mock_energy_monitor.get_remaining_energy.return_value = remaining_energy
            self.mock_energy_monitor.energy_budget_wh = budget
            
            # Get batch size (which internally calculates the ratio)
            batch_size = self.batcher.get_adaptive_batch_size()
            
            # Verify the batch size reflects the energy level appropriately
            if expected_ratio > 0.7:
                # High energy should result in larger batches
                self.assertGreaterEqual(batch_size, self.base_batch_size)
            elif expected_ratio < 0.3:
                # Low energy should result in smaller batches
                self.assertLess(batch_size, self.base_batch_size)
    
    def test_convergence_progress_update(self):
        """Test training progress tracking"""
        initial_progress = self.batcher.training_progress
        
        self.batcher.get_adaptive_batch_size(convergence_progress=0.5)
        
        # Progress should be updated
        self.assertEqual(self.batcher.training_progress, 0.5)
        self.assertNotEqual(self.batcher.training_progress, initial_progress)
    
    def test_zero_energy_scenario(self):
        """Test behavior when no energy remains"""
        self.mock_energy_monitor.get_remaining_energy.return_value = 0.0
        
        batch_size = self.batcher.get_adaptive_batch_size()
        
        # Should still return minimum batch size
        self.assertEqual(batch_size, self.min_batch_size)
    
    def test_negative_energy_scenario(self):
        """Test behavior with negative remaining energy"""
        self.mock_energy_monitor.get_remaining_energy.return_value = -10.0
        
        batch_size = self.batcher.get_adaptive_batch_size()
        
        # Should handle gracefully and return minimum
        self.assertEqual(batch_size, self.min_batch_size)


class TestBatcherEdgeCases(unittest.TestCase):
    """Test edge cases for EnergyAwareBatcher"""
    
    def setUp(self):
        """Set up edge case test fixtures"""
        self.mock_energy_monitor = Mock()
        self.mock_energy_monitor.energy_budget_wh = 100.0
    
    def test_zero_base_batch_size(self):
        """Test with zero base batch size"""
        self.mock_energy_monitor.get_remaining_energy.return_value = 50.0
        
        batcher = EnergyAwareBatcher(
            base_batch_size=0,
            energy_monitor=self.mock_energy_monitor,
            min_batch_size=1
        )
        
        batch_size = batcher.get_adaptive_batch_size()
        self.assertEqual(batch_size, 1)  # Should default to minimum
    
    def test_large_base_batch_size(self):
        """Test with very large base batch size"""
        self.mock_energy_monitor.get_remaining_energy.return_value = 90.0
        
        batcher = EnergyAwareBatcher(
            base_batch_size=1000,
            energy_monitor=self.mock_energy_monitor,
            min_batch_size=1
        )
        
        batch_size = batcher.get_adaptive_batch_size()
        # Should scale appropriately even with large base size
        self.assertGreater(batch_size, 1000)  # High energy should increase it further
    
    def test_min_batch_size_larger_than_base(self):
        """Test when minimum batch size is larger than base batch size"""
        self.mock_energy_monitor.get_remaining_energy.return_value = 10.0  # Low energy
        
        batcher = EnergyAwareBatcher(
            base_batch_size=8,
            energy_monitor=self.mock_energy_monitor,
            min_batch_size=16  # Larger than base
        )
        
        batch_size = batcher.get_adaptive_batch_size()
        # Should respect minimum even if it's larger than calculated size
        self.assertGreaterEqual(batch_size, 16)


class TestBatcherIntegration(unittest.TestCase):
    """Integration tests for adaptive batching"""
    
    def test_realistic_training_scenario(self):
        """Test a realistic training scenario with changing energy"""
        mock_energy_monitor = Mock()
        mock_energy_monitor.energy_budget_wh = 100.0
        
        batcher = EnergyAwareBatcher(
            base_batch_size=32,
            energy_monitor=mock_energy_monitor,
            min_batch_size=2
        )
        
        # Simulate training progress with decreasing energy
        scenarios = [
            (100.0, 0.0, "start"),    # Full energy, start of training
            (75.0, 0.3, "early"),     # 75% energy, early training
            (45.0, 0.6, "middle"),    # 45% energy, middle training
            (20.0, 0.8, "late"),      # 20% energy, late training
            (5.0, 0.95, "final"),     # 5% energy, final phase
        ]
        
        batch_sizes = []
        for remaining_energy, progress, phase in scenarios:
            mock_energy_monitor.get_remaining_energy.return_value = remaining_energy
            
            batch_size = batcher.get_adaptive_batch_size(convergence_progress=progress)
            batch_sizes.append((phase, batch_size))
        
        # Verify batch sizes generally decrease as energy depletes
        start_size = batch_sizes[0][1]
        final_size = batch_sizes[-1][1]
        
        self.assertGreaterEqual(start_size, final_size)
        
        # All batch sizes should be valid
        for phase, size in batch_sizes:
            self.assertGreaterEqual(size, 2)  # >= min_batch_size
            self.assertIsInstance(size, int)


if __name__ == '__main__':
    unittest.main()