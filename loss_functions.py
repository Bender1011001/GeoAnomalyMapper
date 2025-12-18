import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureGuidedTVLoss(nn.Module):
    """
    Structure-Guided Total Variation Loss.
    Encourages edges in the density model to align with edges in the magnetic data.
    """
    def __init__(self):
        super().__init__()

    def forward(self, density, weights):
        """
        density: (B, 1, H, W)
        weights: (B, 1, H, W) - Edge weights derived from magnetic data.
                 Low weight where magnetic edge exists (allow density change).
                 High weight where smooth (enforce smoothness).
        """
        # Calculate gradients of density
        dy = torch.abs(density[:, :, 1:, :] - density[:, :, :-1, :])
        dx = torch.abs(density[:, :, :, 1:] - density[:, :, :, :-1])

        # Pad to match shape (H, W)
        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))

        # Weighted TV
        loss = torch.mean(weights * (dx + dy))
        return loss

def calculate_weights_from_magnetic_gradient(magnetic, beta=1.0):
    """
    Calculates weights for Structure-Guided TV loss.
    High magnetic gradient -> Low weight (allow density boundary).
    Low magnetic gradient -> High weight (enforce smoothness).
    """
    # Calculate gradient magnitude of magnetic field
    dy = torch.abs(magnetic[:, :, 1:, :] - magnetic[:, :, :-1, :])
    dx = torch.abs(magnetic[:, :, :, 1:] - magnetic[:, :, :, :-1])
    
    dy = F.pad(dy, (0, 0, 0, 1))
    dx = F.pad(dx, (0, 1, 0, 0))
    
    grad_mag = torch.sqrt(dx**2 + dy**2 + 1e-8)
    
    # Normalize gradient for stability? Optional, but here we trust input scaling.
    
    # Inverse weighting
    weights = torch.exp(-beta * grad_mag)
    return weights
