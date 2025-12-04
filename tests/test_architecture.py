import unittest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rssm_architecture import HealthcareRSSM

class TestRSSMArchitecture(unittest.TestCase):
    def setUp(self):
        self.model = HealthcareRSSM(
            individual_input_dim=15,
            system_input_dim=10,
            individual_latent_dim=32,
            system_latent_dim=16,
            action_dim=5
        )

    def test_model_initialization(self):
        """Test if model initializes with correct dimensions"""
        self.assertIsInstance(self.model, HealthcareRSSM)
        self.assertEqual(self.model.individual_latent_dim, 32)
        self.assertEqual(self.model.system_latent_dim, 16)

    def test_forward_pass_shapes(self):
        """Test forward pass tensor shapes"""
        batch_size = 4
        seq_len = 5
        
        # Mock inputs
        ind_obs = torch.randn(batch_size, seq_len, 15)
        sys_obs = torch.randn(batch_size, seq_len, 10)
        action = torch.randn(batch_size, 5)
        
        # Test encode
        posterior = self.model.encode(ind_obs, sys_obs)
        self.assertEqual(posterior['z_individual'].shape, (batch_size, 32))
        self.assertEqual(posterior['z_system'].shape, (batch_size, 16))
        
        # Test transition
        prior = self.model.transition_step(
            posterior['z_individual'], 
            posterior['z_system'], 
            action
        )
        self.assertEqual(prior['z_next_individual'].shape, (batch_size, 32))
        self.assertEqual(prior['z_next_system'].shape, (batch_size, 16))

if __name__ == '__main__':
    unittest.main()
