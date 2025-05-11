from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from fourier_nn import train_model, sine_function, complex_function, square_wave, custom_function, FourierNN
from market_data import get_data_for_asset, get_market_data_provider
# Import the new backtesting module
from backtesting import run_backtest, get_backtest_results, NaNEncoder, save_backtest_results
import math

app = Flask(__name__, static_folder='static')

# Initialize neural network and data storage
model = FourierNN()
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(static_dir, exist_ok=True)

# Dictionary of available functions
FUNCTIONS = {
    'sine': sine_function,
    'complex': complex_function,
    'square': square_wave,
    'custom': custom_function
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        function_name = data.get('function', 'sine')
        epochs = int(data.get('epochs', 1000))
        learning_rate = float(data.get('learning_rate', 0.001))
        
        # Get hidden layer configuration from request or use default
        hidden_sizes = data.get('hidden_sizes', [64, 64])
        
        # Get initialization preference
        initialize_with_average = data.get('initialize_with_average', True)
        
        # Get the target function
        target_fn = FUNCTIONS.get(function_name, sine_function)
        
        # Train the model
        model, training_data = train_model(
            target_fn=target_fn,
            num_epochs=epochs,
            learning_rate=learning_rate,
            hidden_sizes=hidden_sizes,
            initialize_with_average=initialize_with_average
        )
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_data')
def get_data():
    try:
        data_file = os.path.join(static_dir, 'data.json')
        with open(data_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': 'No data available. Please train a model first.'})
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return jsonify({'error': str(e)}), 500

# New routes for market data

@app.route('/available_assets')
def available_assets():
    """Get available assets for the UI"""
    provider = get_market_data_provider()
    return jsonify(provider.get_available_assets())

@app.route('/available_timeframes')
def available_timeframes():
    """Get available timeframes for the UI"""
    provider = get_market_data_provider()
    return jsonify(provider.get_available_timeframes())

@app.route('/market_data', methods=['POST'])
def market_data():
    """Fetch market data based on user parameters"""
    try:
        data = request.json
        symbol = data.get('symbol', 'BTC/USDT')
        timeframe = data.get('timeframe', '1d')
        ma_periods = data.get('ma_periods', [20, 50, 200])
        limit = int(data.get('limit', 1000))
        
        # Process and save market data
        result = get_data_for_asset(
            symbol=symbol,
            ma_periods=ma_periods,
            timeframe=timeframe,
            limit=limit
        )
        
        return app.response_class(
            response=json.dumps(result, cls=NaNEncoder),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_market_data')
def get_market_data():
    """Get market data from the server."""
    try:
        market_data_file = os.path.join(static_dir, 'market_data.json')
        if not os.path.exists(market_data_file):
            return jsonify({"error": "No market data found. Please fetch data first."})
            
        with open(market_data_file, 'r') as f:
            market_data = json.load(f)
        
        # Validate the market data
        if not market_data or not isinstance(market_data, dict):
            return jsonify({"error": "Invalid market data format."})
        
        required_fields = ['symbol', 'timeframe', 'dates', 'prices']
        missing_fields = [field for field in required_fields if field not in market_data]
        if missing_fields:
            return jsonify({"error": f"Market data is missing required fields: {', '.join(missing_fields)}"})
        
        # Ensure data has proper types and no invalid values
        sanitized_data = sanitize_data(market_data)
        
        return jsonify(sanitized_data)
    except Exception as e:
        print(f"Error getting market data: {e}")
        return jsonify({"error": f"Failed to get market data: {str(e)}"})

def sanitize_data(data):
    """Sanitize data to ensure it can be properly JSON serialized and handled by the frontend.
    
    Args:
        data: The data structure to sanitize
        
    Returns:
        The sanitized data structure
    """
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    elif isinstance(data, float):
        # Handle special float values
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif isinstance(data, np.ndarray):
        # Convert numpy arrays to lists and sanitize
        return sanitize_data(data.tolist())
    elif isinstance(data, (np.int64, np.int32, np.float64, np.float32)):
        # Convert numpy scalar types to Python native types
        return data.item()
    else:
        return data

# New routes for moving average neural network analysis

@app.route('/train_ma_nn', methods=['POST'])
def train_ma_nn():
    """Train neural network on moving average data"""
    try:
        data = request.json
        ma_period = int(data.get('ma_period', 20))
        epochs = int(data.get('epochs', 1000))
        learning_rate = float(data.get('learning_rate', 0.001))
        
        # Get hidden layer configuration from request or use default
        hidden_sizes = data.get('hidden_sizes', [64, 64])
        
        # Get initialization preference
        initialize_with_average = data.get('initialize_with_average', True)
        
        print(f"Training MA neural network: period={ma_period}, epochs={epochs}, learning_rate={learning_rate}")
        
        # Load market data
        market_data_file = os.path.join(static_dir, 'market_data.json')
        with open(market_data_file, 'r') as f:
            market_data = json.load(f)
            
        # Check if selected MA exists
        if str(ma_period) not in market_data['moving_averages']:
            return jsonify({
                'status': 'error', 
                'message': f'Moving average {ma_period} not found in market data'
            }), 400
            
        # Get MA values
        ma_values = market_data['moving_averages'][str(ma_period)]
        
        # Check if we have enough data points for the neural network
        if not ma_values or len(ma_values) < 10:
            return jsonify({
                'status': 'error',
                'message': f'Insufficient data points for MA {ma_period}. Need at least 10 points.'
            }), 400
        
        # Filter out any None values
        ma_values = [v if v is not None else 0.0 for v in ma_values]
        
        # Create a simple function that takes an index and returns the MA value
        def ma_function(x):
            # Map x from [-10, 10] to [0, len(ma_values)]
            idx = int((x + 10) / 20 * len(ma_values))
            idx = max(0, min(idx, len(ma_values) - 1))  # Ensure index is in range
            return float(ma_values[idx])  # Ensure we return a float
        
        # Train the model with error handling
        try:
            # Test the function first to ensure it works
            test_x = torch.tensor([0.0], dtype=torch.float32)
            test_result = ma_function(test_x.item())
            if not isinstance(test_result, (int, float)) or np.isnan(test_result):
                raise ValueError(f"Invalid value from MA function: {test_result}")
                
            # Now train the model
            model, training_data = train_model(
                target_fn=ma_function,
                num_epochs=epochs,
                learning_rate=learning_rate,
                hidden_sizes=hidden_sizes,
                initialize_with_average=initialize_with_average
            )
            
            # Create normalized indices for x_data
            x_indices = np.linspace(-10, 10, len(ma_values))
            
            # Get final prediction with error handling
            y_pred = []
            for x in x_indices:
                try:
                    pred = model(torch.tensor([x], dtype=torch.float32)).item()
                    y_pred.append(float(pred))  # Ensure we have a float
                except Exception as e:
                    print(f"Error getting prediction for x={x}: {e}")
                    y_pred.append(0.0)  # Use a default value on error
            
            # Save MA neural network data
            ma_nn_data = {
                'ma_period': ma_period,
                'ma_values': ma_values,
                'nn_pred': y_pred,
                'training_data': training_data
            }
            
            # Use the NaN encoder to avoid issues with NaN values
            ma_nn_file = os.path.join(static_dir, 'ma_nn_data.json')
            with open(ma_nn_file, 'w') as f:
                json.dump(ma_nn_data, f, cls=NaNEncoder)
            
            return jsonify({'status': 'success'})
        except Exception as e:
            print(f"Error during MA neural network training: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'status': 'error', 'message': str(e)}), 500
    except FileNotFoundError:
        return jsonify({
            'status': 'error', 
            'message': 'No market data available. Please fetch market data first.'
        }), 400
    except Exception as e:
        print(f"Error training MA neural network: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_ma_nn_data')
def get_ma_nn_data():
    """Get saved MA neural network data"""
    try:
        data_file = os.path.join(static_dir, 'ma_nn_data.json')
        with open(data_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': 'No MA neural network data available. Please train a model first.'})
    except Exception as e:
        print(f"Error retrieving MA neural network data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/run_backtest', methods=['POST'])
def backtest_route():
    """Run a backtest of a trading strategy based on the neural network predictions"""
    try:
        # Get backtest parameters
        params = request.json
        
        # Validate parameters
        required_params = ['initial_capital', 'position_size_pct', 'commission_pct', 'slippage_pct']
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            return jsonify({
                'status': 'error',
                'message': f'Missing required parameters: {", ".join(missing_params)}'
            }), 400
            
        # Validate parameter values
        if params.get('initial_capital', 0) <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Initial capital must be greater than zero'
            }), 400
            
        if not (0 < params.get('position_size_pct', 0) <= 100):
            return jsonify({
                'status': 'error',
                'message': 'Position size percentage must be between 1 and 100'
            }), 400
            
        if params.get('commission_pct', 0) < 0:
            return jsonify({
                'status': 'error',
                'message': 'Commission percentage cannot be negative'
            }), 400
            
        if params.get('slippage_pct', 0) < 0:
            return jsonify({
                'status': 'error',
                'message': 'Slippage percentage cannot be negative'
            }), 400
        
        # Verify required data files exist
        nn_data_file = os.path.join(static_dir, 'ma_nn_data.json')
        market_data_file = os.path.join(static_dir, 'market_data.json')
        
        if not os.path.exists(nn_data_file):
            return jsonify({
                'status': 'error',
                'message': 'Neural network data not found. Please train a model first.'
            }), 400
            
        if not os.path.exists(market_data_file):
            return jsonify({
                'status': 'error',
                'message': 'Market data not found. Please fetch market data first.'
            }), 400
        
        # Load neural network data
        try:
            with open(nn_data_file, 'r') as f:
                nn_data = json.load(f)
        except json.JSONDecodeError as e:
            return jsonify({
                'status': 'error',
                'message': f'Error parsing neural network data: {str(e)}'
            }), 500
        
        # Load market data
        try:
            with open(market_data_file, 'r') as f:
                market_data = json.load(f)
        except json.JSONDecodeError as e:
            return jsonify({
                'status': 'error',
                'message': f'Error parsing market data: {str(e)}'
            }), 500
        
        # Run backtest using the imported function
        backtest_results = run_backtest(
            nn_data=nn_data, 
            market_data=market_data, 
            params=params, 
            static_dir=static_dir
        )
        
        # Save results explicitly
        save_success = save_backtest_results(backtest_results, static_dir)
        if not save_success:
            print("Warning: Failed to save backtest results")
        
        return jsonify({'status': 'success', 'data': backtest_results})
    
    except FileNotFoundError as e:
        return jsonify({
            'status': 'error',
            'message': f'Required data file not found: {str(e)}'
        }), 400
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid input: {str(e)}'
        }), 400
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_backtest_results')
def backtest_results_route():
    """Get backtest results from the server."""
    try:
        backtest_file = os.path.join(static_dir, 'backtest_results.json')
        if not os.path.exists(backtest_file):
            return jsonify({"error": "No backtest results found. Please run a backtest first."})
            
        with open(backtest_file, 'r') as f:
            results = json.load(f)
        
        # Validate the backtest results
        if not results or not isinstance(results, dict):
            return jsonify({"error": "Invalid backtest results format."})
        
        required_fields = ['initial_capital', 'final_capital', 'equity_curve', 'trades']
        missing_fields = [field for field in required_fields if field not in results]
        if missing_fields:
            return jsonify({"error": f"Backtest results are missing required fields: {', '.join(missing_fields)}"})
        
        # Ensure data has proper types and no invalid values
        sanitized_results = sanitize_data(results)
        
        return jsonify(sanitized_results)
    except Exception as e:
        print(f"Error getting backtest results: {e}")
        return jsonify({"error": f"Failed to get backtest results: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True) 