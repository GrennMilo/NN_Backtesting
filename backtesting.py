import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a custom JSON encoder to handle NaN and Infinity values
class NaNEncoder(json.JSONEncoder):
    def default(self, obj):
        """Handle NaN, Infinity, and -Infinity values in JSON serialization.
        
        This is necessary because JSON doesn't support these special numeric values.
        We convert them to strings that can be parsed by JavaScript.
        """
        if isinstance(obj, float):
            if math.isnan(obj):
                return None  # Convert NaN to null for better JS handling
            elif math.isinf(obj):
                if obj > 0:
                    return "Infinity"  # Positive infinity
                else:
                    return "-Infinity"  # Negative infinity
        return super().default(obj)

class Backtester:
    """
    A class to backtest trading strategies based on neural network predictions or other signals.
    Handles the trading logic, performance calculations, and result storage.
    """
    
    def __init__(self, static_dir=None):
        """
        Initialize the backtester.
        
        Args:
            static_dir (str, optional): Directory to save backtest results. If None, will use 'static' in current dir.
        """
        if static_dir is None:
            # Use static directory in the current directory
            self.static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        else:
            self.static_dir = static_dir
            
        # Ensure static directory exists
        os.makedirs(self.static_dir, exist_ok=True)
        
        # Set path for results file
        self.results_file = os.path.join(self.static_dir, 'backtest_results.json')
        
    def run_backtest(self, nn_data, market_data, params):
        """
        Run a backtest using neural network predictions as trading signals.
        
        Args:
            nn_data (dict): Neural network data containing predictions
            market_data (dict): Market data with prices and dates
            params (dict): Backtest parameters (initial_capital, commission_pct, etc.)
            
        Returns:
            dict: Backtest results including performance metrics and trade details
        """
        try:
            logger.info(f"Starting backtest with initial capital: {params['initial_capital']}")
            
            # Extract parameters
            initial_capital = params.get('initial_capital', 10000)
            commission_pct = params.get('commission_pct', 0.1) / 100  # Convert to decimal
            slippage_pct = params.get('slippage_pct', 0.05) / 100  # Convert to decimal
            position_size_pct = params.get('position_size_pct', 100) / 100  # Convert to decimal
            
            # Extract data
            prices = market_data['prices']
            dates = market_data['dates']
            
            # Get neural network predictions
            nn_predictions = nn_data['nn_pred']
            
            # Calculate slopes between consecutive points
            slopes = []
            for i in range(1, len(nn_predictions)):
                slope = nn_predictions[i] - nn_predictions[i-1]
                slopes.append(slope)
            
            # Padding for the first point (no slope)
            slopes.insert(0, 0)
            
            # Initialize backtest variables
            capital = initial_capital
            position = 0  # 0 = no position, 1 = long, -1 = short
            position_size = 0
            entry_price = 0
            entry_date = None
            trades = []
            equity_curve = [capital]
            positions = [0]  # Track positions for each data point
            
            logger.info(f"Running backtest on {len(prices)} data points")
            
            # Run the backtest
            for i in range(1, len(prices)):
                current_price = prices[i]
                previous_price = prices[i-1]
                
                # Calculate slope change
                current_slope = slopes[i]
                previous_slope = slopes[i-1]
                
                # Track equity before any trades
                if position == 0:
                    equity_curve.append(capital)
                elif position == 1:  # Long position
                    current_value = position_size * (current_price / entry_price)
                    equity_curve.append(capital - position_size + current_value)
                elif position == -1:  # Short position
                    current_value = position_size * (2 - current_price / entry_price)
                    equity_curve.append(capital - position_size + current_value)
                
                # Trading logic: slope changes from negative to positive = buy signal
                if previous_slope < 0 and current_slope > 0:
                    # Close any existing short positions
                    if position == -1:
                        # Calculate profit/loss from short
                        pnl = position_size * (entry_price - current_price) / entry_price
                        # Apply commission and slippage
                        commission = position_size * commission_pct
                        slippage = position_size * slippage_pct
                        net_pnl = pnl - commission - slippage
                        capital += net_pnl
                        
                        # Record trade
                        exit_date = dates[i]
                        trades.append({
                            'type': 'short',
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': exit_date,
                            'exit_price': current_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'net_pnl': net_pnl,
                            'commission': commission,
                            'slippage': slippage
                        })
                    
                    # Enter long position
                    position = 1
                    position_size = capital * position_size_pct
                    entry_price = current_price
                    entry_date = dates[i]
                
                # Trading logic: slope changes from positive to negative = sell signal
                elif previous_slope > 0 and current_slope < 0:
                    # Close any existing long positions
                    if position == 1:
                        # Calculate profit/loss from long
                        pnl = position_size * (current_price - entry_price) / entry_price
                        # Apply commission and slippage
                        commission = position_size * commission_pct
                        slippage = position_size * slippage_pct
                        net_pnl = pnl - commission - slippage
                        capital += net_pnl
                        
                        # Record trade
                        exit_date = dates[i]
                        trades.append({
                            'type': 'long',
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': exit_date,
                            'exit_price': current_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'net_pnl': net_pnl,
                            'commission': commission,
                            'slippage': slippage
                        })
                    
                    # Enter short position
                    position = -1
                    position_size = capital * position_size_pct
                    entry_price = current_price
                    entry_date = dates[i]
                
                # Track current position
                positions.append(position)
            
            # Close any open positions at the end of the backtest
            final_price = prices[-1]
            if position == 1:  # Long position
                # Calculate profit/loss from long
                pnl = position_size * (final_price - entry_price) / entry_price
                # Apply commission and slippage
                commission = position_size * commission_pct
                slippage = position_size * slippage_pct
                net_pnl = pnl - commission - slippage
                capital += net_pnl
                
                # Record trade
                exit_date = dates[-1]
                trades.append({
                    'type': 'long',
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': final_price,
                    'position_size': position_size,
                    'pnl': pnl,
                    'net_pnl': net_pnl,
                    'commission': commission,
                    'slippage': slippage
                })
            elif position == -1:  # Short position
                # Calculate profit/loss from short
                pnl = position_size * (entry_price - final_price) / entry_price
                # Apply commission and slippage
                commission = position_size * commission_pct
                slippage = position_size * slippage_pct
                net_pnl = pnl - commission - slippage
                capital += net_pnl
                
                # Record trade
                exit_date = dates[-1]
                trades.append({
                    'type': 'short',
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': final_price,
                    'position_size': position_size,
                    'pnl': pnl,
                    'net_pnl': net_pnl,
                    'commission': commission,
                    'slippage': slippage
                })
            
            # Calculate backtest statistics
            backtest_results = self.calculate_statistics(
                trades=trades,
                equity_curve=equity_curve,
                positions=positions,
                slopes=slopes,
                prices=prices,
                dates=dates,
                initial_capital=initial_capital,
                final_capital=capital
            )
            
            # Save backtest results
            self.save_results(backtest_results)
            
            logger.info(f"Backtest completed with final capital: {capital:.2f}")
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def calculate_statistics(self, trades, equity_curve, positions, slopes, prices, dates, initial_capital, final_capital):
        """
        Calculate performance statistics from backtest results.
        
        Args:
            trades (list): List of trade dictionaries
            equity_curve (list): Equity values over time
            positions (list): Position values over time (0, 1, -1)
            slopes (list): Neural network prediction slopes
            prices (list): Price data
            dates (list): Date strings
            initial_capital (float): Starting capital
            final_capital (float): Ending capital
            
        Returns:
            dict: Statistics and backtest results
        """
        # Calculate basic return metrics
        total_return = (final_capital / initial_capital - 1) * 100
        
        # Calculate drawdowns
        max_equity = initial_capital
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for equity in equity_curve:
            if equity > max_equity:
                max_equity = equity
            drawdown = max_equity - equity
            drawdown_pct = drawdown / max_equity * 100
            if drawdown_pct > max_drawdown_pct:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # Calculate win/loss statistics
        winning_trades = [t for t in trades if t['net_pnl'] > 0]
        losing_trades = [t for t in trades if t['net_pnl'] <= 0]
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        total_trades = len(trades)
        
        win_rate = win_count / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = sum(t['net_pnl'] for t in winning_trades) / win_count if win_count > 0 else 0
        avg_loss = sum(t['net_pnl'] for t in losing_trades) / loss_count if loss_count > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(t['net_pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['net_pnl'] for t in losing_trades))
        
        # Handle case when there are no losing trades
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            # Use a high value to indicate excellent performance rather than infinity
            profit_factor = 999.99 if gross_profit > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(equity_curve)):
            daily_return = (equity_curve[i] / equity_curve[i-1]) - 1
            returns.append(daily_return)
        
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns else 1
        
        # Avoid division by very small numbers
        if std_return > 0.0001:
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252)  # Annualized
        else:
            # If standard deviation is essentially zero, either returns are constant or there's only one return
            # A high Sharpe ratio if returns are positive, otherwise zero
            sharpe_ratio = 999.99 if avg_return > 0 else 0
        
        # Calculate additional metrics
        if total_trades > 0:
            avg_trade = (final_capital - initial_capital) / total_trades
            pct_profitable = win_count / total_trades * 100
        else:
            avg_trade = 0
            pct_profitable = 0
            
        # Calculate max consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for t in trades:
            if t['net_pnl'] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Compile results
        backtest_results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade': avg_trade,
            'pct_profitable': pct_profitable,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'equity_curve': equity_curve,
            'trades': trades,
            'positions': positions,
            'slopes': slopes,
            'prices': prices,
            'dates': dates
        }
        
        return backtest_results
    
    def save_results(self, results):
        """Save backtest results to a JSON file.
        
        Args:
            results (dict): Dictionary of backtest results
        """
        if self.static_dir is None:
            print("Warning: No static directory provided. Results will not be saved.")
            return

        # Ensure results are properly sanitized
        sanitized_results = self._sanitize_data(results)
        
        file_path = os.path.join(self.static_dir, "backtest_results.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(sanitized_results, f, cls=NaNEncoder)
            print(f"Backtest results saved to {file_path}")
        except Exception as e:
            print(f"Error saving backtest results: {e}")
            
    def _sanitize_data(self, data):
        """Sanitize data to ensure it can be properly JSON serialized.
        
        This function recursively processes the data structure to handle special values.
        
        Args:
            data: The data structure to sanitize
            
        Returns:
            The sanitized data structure
        """
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, float):
            # Handle special float values
            if math.isnan(data) or math.isinf(data):
                return None
            return data
        elif isinstance(data, np.ndarray):
            # Convert numpy arrays to lists and sanitize
            return self._sanitize_data(data.tolist())
        elif isinstance(data, (np.int64, np.int32, np.float64, np.float32)):
            # Convert numpy scalar types to Python native types
            return data.item()
        else:
            return data
    
    def load_results(self):
        """
        Load saved backtest results from the JSON file.
        
        Returns:
            dict: The loaded backtest results or None if file doesn't exist
        """
        try:
            if not os.path.exists(self.results_file):
                logger.warning(f"Backtest results file not found: {self.results_file}")
                return None
                
            with open(self.results_file, 'r') as f:
                try:
                    results = json.load(f)
                    
                    # Validate basic structure
                    if not isinstance(results, dict):
                        logger.warning(f"Invalid format in {self.results_file}")
                        return None
                        
                    # Check for NaN replacement
                    for key, value in results.items():
                        if isinstance(value, list):
                            # Replace any None values in arrays with NaN for numpy operations
                            results[key] = [np.nan if v is None else v for v in value]
                    
                    return results
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing {self.results_file}: {e}")
                    return None
        except Exception as e:
            logger.error(f"Error loading backtest results: {e}")
            import traceback
            traceback.print_exc()
            return None

# Functions to use in Flask routes
def get_backtester(static_dir=None):
    """
    Get a backtester instance.
    
    Args:
        static_dir (str, optional): Directory to save backtest results
        
    Returns:
        Backtester: A backtester instance
    """
    return Backtester(static_dir=static_dir)

def run_backtest(nn_data, market_data, params, static_dir=None):
    """
    Run a backtest with the given data and parameters.
    
    Args:
        nn_data (dict): Neural network data
        market_data (dict): Market price data
        params (dict): Backtest parameters
        static_dir (str, optional): Directory to save results
        
    Returns:
        dict: Backtest results
    """
    backtester = get_backtester(static_dir=static_dir)
    return backtester.run_backtest(nn_data, market_data, params)

def get_backtest_results(static_dir):
    """Get saved backtest results"""
    try:
        backtest_file = os.path.join(static_dir, 'backtest_results.json')
        if not os.path.exists(backtest_file):
            print(f"Backtest results file not found: {backtest_file}")
            return None
            
        with open(backtest_file, 'r') as f:
            try:
                results = json.load(f)
                
                # Validate basic structure
                if not isinstance(results, dict):
                    print(f"Warning: Invalid format in backtest_results.json")
                    return None
                    
                # Check for NaN replacement
                for key, value in results.items():
                    if isinstance(value, list):
                        # Replace any None values in arrays with NaN for numpy operations
                        results[key] = [np.nan if v is None else v for v in value]
                
                return results
            except json.JSONDecodeError as e:
                print(f"Error parsing backtest_results.json: {e}")
                return None
    except Exception as e:
        print(f"Error loading backtest results: {e}")
        import traceback
        traceback.print_exc()
        return None

# Save backtest results to a JSON file
def save_backtest_results(results, static_dir):
    """Save backtest results to a JSON file.
    
    Args:
        results (dict): Dictionary of backtest results
        static_dir (str): Directory to save the results
    """
    if static_dir is None:
        print("Warning: No static directory provided. Results will not be saved.")
        return

    file_path = os.path.join(static_dir, "backtest_results.json")
    
    # Sanitize results before saving
    sanitized_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            # Convert numpy arrays to lists
            sanitized_results[key] = value.tolist()
        elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            # Handle lists of numpy arrays
            sanitized_results[key] = [arr.tolist() for arr in value]
        else:
            sanitized_results[key] = value
    
    try:
        with open(file_path, 'w') as f:
            json.dump(sanitized_results, f, cls=NaNEncoder)
        print(f"Backtest results saved to {file_path}")
    except Exception as e:
        print(f"Error saving backtest results: {e}")
        
    return file_path

# Example usage
if __name__ == "__main__":
    # This shows how to use the backtester directly
    backtester = Backtester()
    
    # Load test data (in real usage, this would come from neural network and market data)
    import random
    
    # Mock data
    dates = [f"2023-01-{i:02d}" for i in range(1, 31)]
    prices = [100 + random.normalvariate(0, 1) * 10 for _ in range(30)]
    nn_pred = [price + random.normalvariate(0, 1) * 5 for price in prices]
    
    nn_data = {'nn_pred': nn_pred}
    market_data = {'prices': prices, 'dates': dates}
    params = {'initial_capital': 10000, 'commission_pct': 0.1, 'slippage_pct': 0.05, 'position_size_pct': 100}
    
    # Run backtest
    results = backtester.run_backtest(nn_data, market_data, params)
    print(f"Backtest completed with final capital: {results['final_capital']:.2f}") 