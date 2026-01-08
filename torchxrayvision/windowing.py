import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_random_window_width(img, min_width, max_width=256):
    width = np.random.randint(min_width, max_width + 1)
    img = apply_window(img, 256. / 2, width, y_min=0, y_max=255)
    return img


def apply_window(arr, center, width, y_min=0, y_max=255):
    y_range = y_max - y_min
    arr = arr.astype('float64')
    width = float(width)

    below = arr <= (center - width / 2)
    above = arr > (center + width / 2)
    between = np.logical_and(~below, ~above)

    arr[below] = y_min
    arr[above] = y_max
    if between.any():
        arr[between] = (
                ((arr[between] - center) / width + 0.5) * y_range + y_min
        )

    return arr


class MonotonicSpline(nn.Module):
    """
    Monotonic spline transformation using positive weights to ensure monotonicity.
    """
    def __init__(self, n_knots=10, input_range=(0, 1), init_nonlinear=True):
        super(MonotonicSpline, self).__init__()
        self.n_knots = n_knots
        self.input_min, self.input_max = input_range
        
        # Initialize knot positions uniformly
        self.register_buffer('knot_x', torch.linspace(self.input_min, self.input_max, n_knots))
        
        # Initialize weights with some non-linearity for more interesting transformations
        if init_nonlinear:
            # Create a more dramatic windowing-like initialization
            weights_init = torch.ones(n_knots - 1)
            # Add some variation - create a contrast enhancement curve
            for i in range(n_knots - 1):
                pos = i / (n_knots - 2)  # position from 0 to 1
                # Create a contrast-enhancing S-curve
                # More weight in mid-range, less at extremes (typical windowing)
                if pos < 0.3:
                    weights_init[i] = 0.5  # Compress shadows
                elif pos > 0.7:
                    weights_init[i] = 0.5  # Compress highlights
                else:
                    weights_init[i] = 2.0  # Enhance mid-tones
            
            # Add some randomness for variety
            weights_init += 0.3 * torch.randn(n_knots - 1)
            weights_init = torch.clamp(weights_init, 0.1, 3.0)  # Keep positive and reasonable
            
            self.weights = nn.Parameter(weights_init)
        else:
            # Standard uniform initialization (identity)
            self.weights = nn.Parameter(torch.ones(n_knots - 1))
        
    def forward(self, x):
        """
        Apply monotonic spline transformation to input tensor.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Transformed tensor of same shape as input
        """
        original_shape = x.shape
        x_flat = x.view(-1)
        
        # Ensure weights are positive for monotonicity
        positive_weights = F.softplus(self.weights)
        
        # Create knot y-values by cumulative sum of positive weights
        knot_y = torch.cat([torch.zeros(1, device=x.device), torch.cumsum(positive_weights, dim=0)])
        
        # Normalize to [0, 1] range
        knot_y = knot_y / knot_y[-1]
        
        # Apply piecewise linear interpolation
        y_flat = self._interpolate(x_flat, self.knot_x, knot_y)
        
        return y_flat.view(original_shape)
    
    def _interpolate(self, x, knot_x, knot_y):
        """
        Piecewise linear interpolation between knots.
        """
        # Find the interval for each x value
        indices = torch.searchsorted(knot_x[1:], x, right=False)
        indices = torch.clamp(indices, 0, len(knot_x) - 2)
        
        # Get the surrounding knot points
        x0 = knot_x[indices]
        x1 = knot_x[indices + 1]
        y0 = knot_y[indices]
        y1 = knot_y[indices + 1]
        
        # Linear interpolation
        alpha = (x - x0) / (x1 - x0 + 1e-8)  # Add small epsilon to avoid division by zero
        alpha = torch.clamp(alpha, 0, 1)
        
        return y0 + alpha * (y1 - y0)
    
    def get_spline_params(self):
        """
        Get current spline parameters for visualization.
        
        Returns:
            dict: Dictionary containing knot positions and values
        """
        with torch.no_grad():
            positive_weights = F.softplus(self.weights)
            knot_y = torch.cat([torch.zeros(1, device=self.weights.device), torch.cumsum(positive_weights, dim=0)])
            knot_y = knot_y / knot_y[-1]
            
            return {
                'knot_x': self.knot_x.cpu().numpy(),
                'knot_y': knot_y.cpu().numpy(),
                'weights': positive_weights.cpu().numpy()
            }


class SplineWindowingFunction(nn.Module):
    """
    Learnable windowing function using monotonic splines for chest X-ray preprocessing.
    Designed to remove dataset-specific artifacts while preserving clinical information.
    """
    def __init__(self, n_knots=10, input_range=(0, 1), learnable=True, init_nonlinear=True):
        super(SplineWindowingFunction, self).__init__()
        self.learnable = learnable
        self.input_range = input_range
        
        if learnable:
            self.spline = MonotonicSpline(n_knots=n_knots, input_range=input_range, init_nonlinear=init_nonlinear)
        else:
            # Identity transformation when not learnable
            self.spline = None
            
    def forward(self, x):
        """
        Apply spline windowing to input images.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width) or any shape
            
        Returns:
            Windowed tensor of same shape as input
        """
        if not self.learnable or self.spline is None:
            return x
            
        # Normalize input to spline range if needed
        x_min, x_max = self.input_range
        if x.min() < x_min or x.max() > x_max:
            # Normalize to [0, 1] range
            x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        else:
            x_norm = x
            
        # Apply spline transformation
        x_windowed = self.spline(x_norm)
        
        return x_windowed
    
    def get_windowing_params(self):
        """
        Get windowing parameters for visualization and analysis.
        
        Returns:
            dict: Dictionary containing spline parameters
        """
        if not self.learnable or self.spline is None:
            return {'type': 'identity'}
            
        params = self.spline.get_spline_params()
        params['type'] = 'monotonic_spline'
        params['n_knots'] = self.spline.n_knots
        params['input_range'] = self.input_range
        
        return params
    
    def visualize_mapping(self, n_points=1000):
        """
        Generate points for visualizing the windowing mapping.
        
        Args:
            n_points: Number of points to generate for visualization
            
        Returns:
            tuple: (input_values, output_values) as numpy arrays
        """
        if not self.learnable or self.spline is None:
            x = np.linspace(self.input_range[0], self.input_range[1], n_points)
            return x, x  # Identity mapping
            
        with torch.no_grad():
            # Get device from spline parameters
            device = next(self.spline.parameters()).device
            x = torch.linspace(self.input_range[0], self.input_range[1], n_points, device=device)
            y = self.spline(x)
            return x.cpu().numpy(), y.cpu().numpy()


class WindowingWrapper(nn.Module):
    """
    Wrapper to add windowing preprocessing to any existing model.
    """
    def __init__(self, base_model, windowing_function=None, apply_to_input=True):
        super(WindowingWrapper, self).__init__()
        self.base_model = base_model
        self.windowing_function = windowing_function or SplineWindowingFunction()
        self.apply_to_input = apply_to_input
        
    def forward(self, x):
        """
        Forward pass with optional windowing preprocessing.
        
        Args:
            x: Input tensor
            
        Returns:
            Output from base model after optional windowing
        """
        if self.apply_to_input and self.windowing_function is not None:
            x = self.windowing_function(x)
            
        return self.base_model(x)
    
    def get_windowing_params(self):
        """Get windowing parameters from the windowing function."""
        if self.windowing_function is not None:
            return self.windowing_function.get_windowing_params()
        return {'type': 'none'}
    
    def set_windowing_learnable(self, learnable=True):
        """Enable or disable learning for the windowing function."""
        if hasattr(self.windowing_function, 'learnable'):
            self.windowing_function.learnable = learnable


def create_windowing_model(base_model, n_knots=10, learnable=True, input_range=(0, 1), init_nonlinear=True):
    """
    Factory function to create a model with windowing preprocessing.
    
    Args:
        base_model: The base model to wrap
        n_knots: Number of knots for the monotonic spline
        learnable: Whether the windowing function should be learnable
        input_range: Input range for the spline (min, max)
        init_nonlinear: Whether to initialize with non-linear curve (vs identity)
        
    Returns:
        WindowingWrapper: Model with windowing preprocessing
    """
    windowing_function = SplineWindowingFunction(
        n_knots=n_knots, 
        input_range=input_range, 
        learnable=learnable,
        init_nonlinear=init_nonlinear
    )
    
    return WindowingWrapper(base_model, windowing_function)


def extract_windowing_params_from_checkpoint(checkpoint_path):
    """
    Extract windowing parameters from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        dict: Windowing parameters or None if not found
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Look for windowing parameters in the model state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        windowing_params = {}
        
        # Extract spline weights
        if 'windowing_function.spline.weights' in state_dict:
            windowing_params['weights'] = state_dict['windowing_function.spline.weights'].numpy()
            
        # Extract knot positions
        if 'windowing_function.spline.knot_x' in state_dict:
            windowing_params['knot_x'] = state_dict['windowing_function.spline.knot_x'].numpy()
            
        # Calculate knot y-values
        if 'weights' in windowing_params:
            positive_weights = F.softplus(torch.tensor(windowing_params['weights']))
            knot_y = torch.cat([torch.zeros(1), torch.cumsum(positive_weights, dim=0)])
            knot_y = knot_y / knot_y[-1]
            windowing_params['knot_y'] = knot_y.numpy()
            windowing_params['type'] = 'monotonic_spline'
            
        return windowing_params if windowing_params else None
        
    except Exception as e:
        print(f"Error extracting windowing parameters: {e}")
        return None
