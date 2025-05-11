import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fft import fft, ifft
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FourierNN(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Neural Network with Fourier Transform capabilities.
    
    The network consists of an input layer, multiple hidden layers, and an output layer.
    Each layer is fully connected (dense) with configurable sizes. ReLU activation
    functions are used between layers to introduce non-linearity.
    
    Attributes:
        layers (nn.ModuleList): List of linear layers in the network
        activation (nn.Module): Activation function used between layers
        history (dict): Dictionary to store training history for visualization
    """
    def __init__(self, input_size=1, hidden_sizes=[64, 64], output_size=1, activation=nn.ReLU(), initial_bias=None):
        """
        Initialize the Multi-Layer Perceptron network.
        
        Args:
            input_size (int): Size of the input features
            hidden_sizes (list): List of integers specifying the size of each hidden layer
            output_size (int): Size of the output
            activation (nn.Module): Activation function to use between layers
            initial_bias (float, optional): Initial bias value for the output layer to start predictions at a specific value
        """
        super(FourierNN, self).__init__()
        
        # Build network architecture
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        
        # Create layers with the specified sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        self.activation = activation
        
        # Initialize the output layer bias if provided
        if initial_bias is not None:
            with torch.no_grad():
                self.layers[-1].bias.fill_(initial_bias)
        
        # Initialize history for visualization
        self.history = {
            'loss': [],
            'predictions': [],
            'fourier_coeffs': []
        }
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output of the network
        """
        # Pass input through each layer with activation
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all but the last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
    
    def compute_fourier(self, x_data):
        """
        Compute Fourier transform of the current model's output.
        
        This method passes the input data through the network, then applies
        Fast Fourier Transform (FFT) to analyze the frequency components.
        
        Args:
            x_data (array-like): Input data points
            
        Returns:
            list: Magnitudes of Fourier coefficients
        """
        try:
            x_tensor = torch.tensor(x_data, dtype=torch.float32).reshape(-1, 1)
            with torch.no_grad():
                y_pred = self(x_tensor).numpy().flatten()
            
            # Check for NaN values
            if np.isnan(y_pred).any():
                logger.warning("NaN values found in model prediction. Replacing with zeros.")
                y_pred = np.nan_to_num(y_pred, nan=0.0)
            
            # Compute FFT
            fft_result = fft(y_pred)
            # Take absolute value to get magnitudes
            fft_mag = np.abs(fft_result)
            # Normalize and take the first half (due to symmetry)
            n = len(fft_mag)
            fft_mag = fft_mag[:n//2] / n
            
            # Convert to list of floats and handle any NaN values
            return [float(x) if not np.isnan(x) else 0.0 for x in fft_mag]
        except Exception as e:
            logger.error(f"Error computing Fourier transform: {e}")
            # Return a default array on error
            return [0.0] * (len(x_data) // 2)

def train_model(target_fn, x_range=(-10, 10), num_points=1000, num_epochs=1000, learning_rate=0.001, hidden_sizes=[64, 64], initialize_with_average=True):
    """
    Train the neural network to approximate the target function
    
    Args:
        target_fn: Function to approximate
        x_range: Range of x values
        num_points: Number of points to sample
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        hidden_sizes: List specifying the size of each hidden layer
        initialize_with_average: Whether to initialize the network with the average target value
        
    Returns:
        model: Trained FourierNN model
        training_data: Dictionary with training history
    """
    try:
        # Create input data
        x_data = np.linspace(x_range[0], x_range[1], num_points)
        
        # Apply target function with error handling
        y_data = []
        for x in x_data:
            try:
                y = target_fn(x)
                # Ensure the result is a valid float
                if not isinstance(y, (int, float)) or np.isnan(y) or np.isinf(y):
                    y = 0.0
                y_data.append(float(y))
            except Exception as e:
                logger.error(f"Error evaluating target function at x={x}: {e}")
                y_data.append(0.0)
        
        y_data = np.array(y_data)
        
        # Calculate average value for initialization if requested
        initial_bias = None
        if initialize_with_average and len(y_data) > 0:
            # Use average of first, last, max, min, and mean for a better starting point
            first_val = y_data[0]
            last_val = y_data[-1]
            mean_val = np.mean(y_data)
            initial_bias = mean_val
            logger.info(f"Initializing neural network with bias: {initial_bias}")
        
        # Convert to tensors
        x_tensor = torch.tensor(x_data, dtype=torch.float32).reshape(-1, 1)
        y_tensor = torch.tensor(y_data, dtype=torch.float32).reshape(-1, 1)
        
        # Create model and optimizer
        model = FourierNN(input_size=1, hidden_sizes=hidden_sizes, output_size=1, initial_bias=initial_bias)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        training_data = {
            'epochs': [],
            'loss': [],
            'predictions': [],
            'fourier_coeffs': []
        }
        
        save_interval = max(1, num_epochs // 50)  # Save ~50 snapshots during training
        
        for epoch in range(num_epochs):
            # Forward pass
            y_pred = model(x_tensor)
            loss = criterion(y_pred, y_tensor)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Save data for visualization at intervals
            if epoch % save_interval == 0 or epoch == num_epochs - 1:
                with torch.no_grad():
                    current_pred = model(x_tensor).numpy().flatten()
                    # Handle NaN values
                    current_pred = np.nan_to_num(current_pred, nan=0.0)
                
                training_data['epochs'].append(int(epoch))
                training_data['loss'].append(float(loss.item()))
                
                if len(training_data['predictions']) < 50:  # Limit stored predictions to avoid huge data
                    training_data['predictions'].append([float(x) for x in current_pred.tolist()])
                    fourier_coeffs = model.compute_fourier(x_data)
                    training_data['fourier_coeffs'].append(fourier_coeffs)
        
        # Final prediction for visualization
        with torch.no_grad():
            final_pred = model(x_tensor).numpy().flatten()
            # Handle NaN values
            final_pred = np.nan_to_num(final_pred, nan=0.0)
        
        # Create sample data for the web interface
        web_data = {
            'x_data': [float(x) for x in x_data.tolist()],
            'y_true': [float(y) for y in y_data.tolist()],
            'y_pred': [float(y) for y in final_pred.tolist()],
            'training_data': training_data,
            'initial_bias': initial_bias
        }
        
        # Ensure static directory exists
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        os.makedirs(static_dir, exist_ok=True)
        
        # Create a JSON encoder that handles NaN values
        class NaNEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                    return 0.0
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                return super(NaNEncoder, self).default(obj)
        
        # Save to JSON with full path
        data_file = os.path.join(static_dir, 'data.json')
        with open(data_file, 'w') as f:
            json.dump(web_data, f, cls=NaNEncoder)
        
        return model, training_data
    
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        import traceback
        traceback.print_exc()
        # Return a minimal model and training data in case of error
        model = FourierNN()
        return model, {
            'epochs': [0],
            'loss': [0.0],
            'predictions': [[]],
            'fourier_coeffs': [[]]
        }

# Example target functions
def sine_function(x):
    return np.sin(x)

def complex_function(x):
    return np.sin(x) + 0.5 * np.sin(2 * x) + 0.3 * np.sin(3 * x)

def square_wave(x):
    return np.sign(np.sin(x))

def custom_function(x):
    return x**2 * np.sin(x) / (1 + abs(x)) 