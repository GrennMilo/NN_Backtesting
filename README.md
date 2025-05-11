# Function Approximation with Neural Networks and Fourier Transform

This application demonstrates three key functionalities:

1. How a neural network can be trained to approximate various mathematical functions with visualization of the Fourier transform
2. Market data analysis with dynamic moving averages for cryptocurrencies and stocks
3. Trading strategy backtesting based on neural network predictions

## Features

### Function Approximation
- Train a neural network to match different mathematical functions
- Visualize the neural network's output compared to the target function
- Display the Fourier transform of the network's output
- Interactive slider to view the progression of learning over time
- Real-time loss visualization

### Market Data Analysis
- Fetch historical price data for cryptocurrencies like Bitcoin
- Calculate and display moving averages with customizable periods
- Support for multiple time frames (minutes, hours, days, weeks)
- Dynamic addition and removal of moving average indicators
- Clean dark-mode visualization interface

### Trading Strategy Backtesting
- Backtest trading strategies based on neural network predictions
- Comprehensive performance metrics (profit/loss, drawdown, win rate, etc.)
- Visualize equity curve and trade entries/exits on price chart
- Configure trading parameters (initial capital, position size, etc.)
- Realistic simulation with commission and slippage costs

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- SciPy
- Flask
- Pandas
- CCXT (for market data)
- Plotly (for interactive charts)

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:

```bash
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:5000`

3. Choose between Function Approximation or Market Data Analysis using the tabs

### For Function Approximation:
- Select a target function, number of epochs, and learning rate
- Click "Train Model" to start the training process
- Use the slider to see how the neural network's approximation evolved during training

### For Market Data Analysis:
- Select an asset (cryptocurrency or stock)
- Choose a timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w)
- Set the number of data points to fetch
- Add or remove moving average periods (2-500 range)
- Click "Fetch Data" to retrieve and display the data

### For Trading Strategy Backtesting:
- First apply a neural network to a moving average
- Configure backtesting parameters:
  - Initial capital: Starting amount to trade with
  - Position size: Percentage of capital to use per trade
  - Commission: Trading fee as a percentage
  - Slippage: Estimated execution cost as a percentage
- Click "Run Backtest" to execute the strategy
- View detailed performance metrics and trade visualizations

## Multi-Layer Neural Network Architecture

This application uses a Multi-Layer Perceptron (MLP) neural network to approximate functions and analyze market data. Here's how it works:

### Key Components

1. **Input Layer**: The first layer of the network that receives the input data (e.g., x-values for function approximation or time indices for market data)

2. **Hidden Layers**: Multiple intermediate layers that transform the data through weighted connections
   - Our implementation allows for configurable number and size of hidden layers
   - Default configuration uses two hidden layers with 64 neurons each

3. **Output Layer**: The final layer that produces the prediction (function value or market data approximation)

4. **Activation Functions**: Non-linear functions applied between layers to enable the network to learn complex patterns
   - We use ReLU (Rectified Linear Unit) activation: `f(x) = max(0, x)`
   - ReLU helps with faster training and addresses the vanishing gradient problem

### How the Network Learns

#### Forward Propagation
During forward propagation, data flows from the input layer through the hidden layers to the output layer:

1. For each neuron in a layer, a weighted sum of inputs is calculated: `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
   - where `w` are weights, `x` are inputs, and `b` is a bias term

2. The activation function is applied to this sum: `a = activation(z)`

3. This process repeats for each layer, with the outputs of one layer becoming the inputs to the next

#### Loss Function
After forward propagation, the network's prediction is compared to the target value using a loss function:

- We use Mean Squared Error (MSE): `MSE = (1/n) * Σ(y_true - y_pred)²`
- This quantifies how far the predictions are from the actual values

#### Backpropagation
Backpropagation is the process of updating the weights to reduce the loss:

1. Compute the gradient of the loss function with respect to each weight
2. Propagate this gradient backwards through the network
3. Update weights using gradient descent: `w_new = w_old - learning_rate * gradient`

#### Optimization
We use the Adam optimizer, which:
- Adapts the learning rate for each parameter based on historical gradients
- Incorporates momentum to help escape local minima
- Provides faster convergence than standard gradient descent

### Fourier Transform Analysis

After training, we apply Fast Fourier Transform (FFT) to the neural network's output to:
1. Analyze the frequency components of the learned function
2. Visualize how well the network captures different frequencies in the data
3. Track how the frequency spectrum evolves during training

This provides insight into how the network is learning to represent different frequency components of the target function.

## Trading Strategy Backtest System

The application includes a comprehensive backtesting system for evaluating trading strategies based on neural network predictions.

### Trading Strategy

The default strategy uses changes in the slope of the neural network's output:

1. When the slope changes from negative to positive:
   - Close any existing short positions
   - Open a long position

2. When the slope changes from positive to negative:
   - Close any existing long positions
   - Open a short position

### Performance Metrics

The backtesting system calculates a wide range of performance metrics:

- **Total Return**: Percentage gain/loss over the backtest period
- **Max Drawdown**: Largest percentage drop from a peak to a trough
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Average Trade**: Average profit/loss per trade
- **Consecutive Wins/Losses**: Maximum streak of winning or losing trades

### Realistic Simulation

The backtester includes important real-world trading costs:

- **Commission**: Trading fees as a percentage of position size
- **Slippage**: Estimated execution cost as a percentage
- **Position Sizing**: Dynamic position sizing based on current equity

### Visualization

Results are visualized in multiple ways:

- **Equity Curve**: Chart showing the growth of account equity over time
- **Trade Markers**: Entry and exit points displayed directly on the price chart
- **Performance Dashboard**: Summary of key performance metrics

## Project Structure

- **app.py**: Flask server and API endpoints
- **fourier_nn.py**: Neural network implementation for function approximation
- **market_data.py**: Market data fetching and processing using CCXT
- **backtesting.py**: Trading strategy backtesting engine
- **static/**: Directory for storing JSON data files
- **templates/**: Frontend HTML and JavaScript

## Available Functions

- **Sine Wave**: Simple sine function
- **Complex Sine**: Combination of sine waves at different frequencies
- **Square Wave**: A square wave function using sign(sin(x))
- **Custom Function**: A more complex function combining polynomial and trigonometric elements 