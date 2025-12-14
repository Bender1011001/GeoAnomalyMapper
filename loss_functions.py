import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureGuidedTVLoss(nn.Module):
    """
    Computes the Structure-Guided (Edge-Weighted) Total Variation Loss.
    
    This loss encourages the model to have sparse gradients (blocky structure),
    but relaxes this penalty at locations where the 'guide_weights' are low.
    
    Mathematical Form: sum( W * |grad(m)| )
    """
    def __init__(self, reduction='sum', epsilon=1e-6):
        super(StructureGuidedTVLoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        
        # Define 3D Finite Difference Kernels (Central Difference or Forward)
        # Using simple forward difference for compactness: [0, -1, 1]
        
        # Kernel for d/dx (width dimension)
        # Shape: (Out_C, In_C/Groups, D, H, W) -> (1, 1, 1, 1, 3)
        k_x = torch.FloatTensor([[[[[0, -1, 1]]]]])
        
        # Kernel for d/dy (height dimension)
        # Shape: (1, 1, 1, 3, 1)
        k_y = torch.FloatTensor([[[[[0],
                                    [-1],
                                    [1]]]]])
        
        # Kernel for d/dz (depth dimension)
        # Shape: (1, 1, 3, 1, 1)
        k_z = torch.FloatTensor([[[[[0]],
                                    [[-1]],
                                    [[1]]]]])
                                  
        # Register kernels as non-trainable buffers
        self.register_buffer('kernel_x', k_x)
        self.register_buffer('kernel_y', k_y)
        self.register_buffer('kernel_z', k_z)

    def forward(self, model_tensor, guide_weights):
        """
        Args:
            model_tensor (Tensor): The 3D Density Model to regularize.
                                   Shape: (Batch, Channel, Depth, Height, Width)
            guide_weights (Tensor): The spatial weight tensor derived from Magnetics/DEM.
                                    Shape: must match model_tensor.
                                    Values: High (1.0) in smooth areas, Low (0.0) at edges.
        
        Returns:
            loss (Tensor): Scalar loss value.
        """
        # Ensure input is 5D
        if len(model_tensor.shape) == 4:
            # Assumes (B, C, H, W) - Treat Depth as 1
            model_tensor = model_tensor.unsqueeze(2)
            if guide_weights is not None and len(guide_weights.shape) == 4:
                guide_weights = guide_weights.unsqueeze(2)

        # 1. Compute Spatial Gradients of the Model
        # Using padding to maintain dimensions
        # Padding: (W_left, W_right, H_top, H_bottom, D_front, D_back)
        
        # Pad for forward difference (last element needs specific padding to validly compute diff, or central)
        # Here we use standard padding=1 for filter size 3
        # Ideally mirror padding or replicate to avoid boundary artifacts
        
        # Note: If kernels are size 3, padding=1 maintains size.
        # But my kernels above are effective size 3 [0, -1, 1] but sparse? 
        # Actually in the plan I wrote [0, -1, 1]. In conv3d it needs to be 3x3x3 usually to be consistent 
        # or specific kernel size.
        # Let's use simple finite diff via conv3d with kernel size 1, 1, 3 etc? No, kernel must be D,H,W
        
        # To match the user's research report exactly, I should use the kernels they defined but correct the shapes if needed.
        # Their research report python code had:
        # k_x = torch.FloatTensor([[[, , ], [, [0, -1, 1], ], [, , ]]]).unsqueeze(0)
        # That is a 3x3x3 kernel.
        
        # Let's stick to a robust implementation using simple slicing which is faster and cleaner for standard TV
        # But since we need "Structure Guided", we need per-pixel weights.
        # Slicing works per pixel too.
        
        diff_z = model_tensor[:, :, 1:, :, :] - model_tensor[:, :, :-1, :, :]
        diff_y = model_tensor[:, :, :, 1:, :] - model_tensor[:, :, :, :-1, :]
        diff_x = model_tensor[:, :, :, :, 1:] - model_tensor[:, :, :, :, :-1]
        
        # Pad differences to match original shape to multiply by weights
        # We assume gradient at edge is 0 or same as neighbor.
        diff_z = F.pad(diff_z, (0,0, 0,0, 0,1))
        diff_y = F.pad(diff_y, (0,0, 0,1, 0,0))
        diff_x = F.pad(diff_x, (0,1, 0,0, 0,0))
        
        # 2. Compute Gradient Magnitude (L2 norm of the vector field)
        # Adding epsilon for numerical stability (Charbonnier approximation)
        grad_mag = torch.sqrt(diff_x**2 + diff_y**2 + diff_z**2 + self.epsilon)
        
        # 3. Apply Structure-Guided Weights
        # The penalty is masked by the guide_weights.
        if guide_weights is not None:
             # Ensure guide_weights match shape if necessary
            if guide_weights.shape != grad_mag.shape:
                guide_weights = F.interpolate(guide_weights, size=grad_mag.shape[2:], mode='nearest')
            
            weighted_tv = guide_weights * grad_mag
        else:
            weighted_tv = grad_mag
        
        # 4. Reduction
        if self.reduction == 'sum':
            return weighted_tv.sum()
        elif self.reduction == 'mean':
            return weighted_tv.mean()
        else:
            return weighted_tv

def calculate_weights_from_magnetic_gradient(mag_grid, beta=1.0, eta=1e-4):
    """
    Generates the Weight Tensor from the Magnetic Horizontal Gradient Magnitude (HGM).
    
    Args:
        mag_grid (Tensor): (B, C, H, W) or (C, H, W) Magnetic Intensity Grid.
        beta (float): Edge sensitivity parameter. Higher = sharper edges.
        eta (float): Stability constant.
    
    Returns:
        weights (Tensor): Normalized weights in range .
    """
    # Ensure mag_grid is tensor
    if not isinstance(mag_grid, torch.Tensor):
        mag_grid = torch.tensor(mag_grid)
        
    device = mag_grid.device
    
    # 1. Calculate Gradient of Magnetic Data
    # Use spatial dimensions (last two)
    dims = (-2, -1)
    
    # torch.gradient returns gradients for specified dims
    grads = torch.gradient(mag_grid, dim=dims)
    
    # Calculate magnitude squared
    grad_sq_sum = torch.zeros_like(mag_grid)
    for g in grads:
        grad_sq_sum += g**2
        
    mag_grad_mag = torch.sqrt(grad_sq_sum)
    
    # 2. Normalize Gradient
    # Normalize per image in batch? Or global batch?
    # Usually global batch or per image. Let's do per image for robustness using keepdim
    min_val = mag_grad_mag.flatten(start_dim=-2).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    max_val = mag_grad_mag.flatten(start_dim=-2).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    
    # Avoid zero div
    denom = max_val - min_val
    denom[denom < 1e-9] = 1.0 # Safety
    
    mag_grad_norm = (mag_grad_mag - min_val) / denom
    
    # 3. Edge Stopping Function (Perona-Malik style)
    # W = 1 / (1 + |grad/K|^beta) where K is a scale param, here simplified
    # The user manual used (grad/eta)^beta
    
    # Avoid division by zero if K/eta is implicit or just using the norm directly
    weights = 1.0 / (1.0 + (mag_grad_norm / eta)**beta)
    
    return weights
