import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Statistical imports
from scipy import stats
from scipy.stats import pearsonr

class ETHVolatilityForecaster:
    """
    ETH Implied Volatility Forecasting Model
    
    This class implements a comprehensive approach to forecasting 10-second ahead
    implied volatility for Ethereum using high-frequency orderbook data and 
    cross-asset signals.
    
    Key Features:
    - Orderbook imbalance and depth features
    - Rolling realized volatility calculations
    - Cross-asset momentum and correlation features
    - Time-series validation with no look-ahead bias
    - Ensemble modeling approach
    """
    
    def __init__(self, data_path_train="train/", data_path_test="test/"):
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test
        self.eth_train = None
        self.eth_test = None
        self.cross_assets = {}
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def load_data(self):
        """Load all training and test data"""
        print("Loading ETH training data...")
        self.eth_train = pd.read_csv(f"{self.data_path_train}ETH.csv")
        self.eth_train['timestamp'] = pd.to_datetime(self.eth_train['timestamp'])
        self.eth_train = self.eth_train.sort_values('timestamp').reset_index(drop=True)
        
        print("Loading ETH test data...")
        self.eth_test = pd.read_csv(f"{self.data_path_test}ETH.csv")
        self.eth_test['timestamp'] = pd.to_datetime(self.eth_test['timestamp'])
        self.eth_test = self.eth_test.sort_values('timestamp').reset_index(drop=True)
        
        # Load cross-asset data for feature engineering
        cross_asset_files = ['BTC.csv', 'SOL.csv']
        for asset_file in cross_asset_files:
            try:
                asset_name = asset_file.replace('.csv', '')
                print(f"Loading {asset_name} training data...")
                self.cross_assets[f"{asset_name}_train"] = pd.read_csv(f"{self.data_path_train}{asset_file}")
                self.cross_assets[f"{asset_name}_train"]['timestamp'] = pd.to_datetime(
                    self.cross_assets[f"{asset_name}_train"]['timestamp'])
                self.cross_assets[f"{asset_name}_train"] = self.cross_assets[f"{asset_name}_train"].sort_values('timestamp').reset_index(drop=True)
            except FileNotFoundError:
                print(f"Warning: {asset_file} not found in training data")
                
        print(f"Loaded data shapes - ETH Train: {self.eth_train.shape}, ETH Test: {self.eth_test.shape}")
        
    def clean_data(self, df):
        """Clean data by handling missing values and outliers"""
        df_clean = df.copy()
        
        # Forward fill missing values for small gaps (up to 5 seconds)
        for col in df_clean.columns:
            if col not in ['timestamp', 'label']:
                df_clean[col] = df_clean[col].fillna(method='ffill', limit=5)
        
        # Handle outliers using percentile clipping (more conservative than IQR)
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'label':  # Don't clip the target variable
                lower_bound = df_clean[col].quantile(0.001)
                upper_bound = df_clean[col].quantile(0.999)
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        return df_clean
    
    def calculate_orderbook_features(self, df, prefix=""):
        """Calculate comprehensive orderbook-based features"""
        df_features = df.copy()
        
        # Basic spread features
        df_features[f'{prefix}spread'] = df_features['ask_price1'] - df_features['bid_price1']
        df_features[f'{prefix}spread_pct'] = df_features[f'{prefix}spread'] / df_features['mid_price']
        
        # Volume imbalance at different levels
        for level in range(1, 6):
            bid_vol = f'bid_volume{level}'
            ask_vol = f'ask_volume{level}'
            df_features[f'{prefix}volume_imbalance_{level}'] = (
                (df_features[bid_vol] - df_features[ask_vol]) / 
                (df_features[bid_vol] + df_features[ask_vol] + 1e-8)
            )
        
        # Total volume features
        bid_volumes = [f'bid_volume{i}' for i in range(1, 6)]
        ask_volumes = [f'ask_volume{i}' for i in range(1, 6)]
        
        df_features[f'{prefix}total_bid_volume'] = df_features[bid_volumes].sum(axis=1)
        df_features[f'{prefix}total_ask_volume'] = df_features[ask_volumes].sum(axis=1)
        df_features[f'{prefix}total_volume'] = df_features[f'{prefix}total_bid_volume'] + df_features[f'{prefix}total_ask_volume']
        df_features[f'{prefix}volume_ratio'] = df_features[f'{prefix}total_bid_volume'] / (df_features[f'{prefix}total_ask_volume'] + 1e-8)
        
        # Volume-weighted prices
        bid_prices = [f'bid_price{i}' for i in range(1, 6)]
        ask_prices = [f'ask_price{i}' for i in range(1, 6)]
        
        df_features[f'{prefix}vwap_bid'] = (df_features[bid_prices] * df_features[bid_volumes]).sum(axis=1) / (df_features[bid_volumes].sum(axis=1) + 1e-8)
        df_features[f'{prefix}vwap_ask'] = (df_features[ask_prices] * df_features[ask_volumes]).sum(axis=1) / (df_features[ask_volumes].sum(axis=1) + 1e-8)
        df_features[f'{prefix}vwap_spread'] = df_features[f'{prefix}vwap_ask'] - df_features[f'{prefix}vwap_bid']
        
        # Price impact and depth features
        df_features[f'{prefix}price_impact_buy'] = (df_features['ask_price5'] - df_features['ask_price1']) / df_features['ask_price1']
        df_features[f'{prefix}price_impact_sell'] = (df_features['bid_price1'] - df_features['bid_price5']) / df_features['bid_price1']
        
        # Microprice (theoretical fair price)
        df_features[f'{prefix}microprice'] = (
            (df_features['bid_price1'] * df_features['ask_volume1'] + 
             df_features['ask_price1'] * df_features['bid_volume1']) / 
            (df_features['bid_volume1'] + df_features['ask_volume1'] + 1e-8)
        )
        
        return df_features
    
    def calculate_time_series_features(self, df, prefix=""):
        """Calculate comprehensive time-series based features"""
        df_features = df.copy()
        
        # Multiple time windows for different signals
        windows = [5, 10, 20, 30, 60, 120, 300]  # 5s to 5min
        
        for window in windows:
            # Price returns and momentum
            df_features[f'{prefix}return_{window}s'] = df_features['mid_price'].pct_change(window)
            df_features[f'{prefix}log_return_{window}s'] = np.log(df_features['mid_price'] / df_features['mid_price'].shift(window))
            
            # Realized volatility (key feature for IV prediction)
            returns = df_features['mid_price'].pct_change()
            df_features[f'{prefix}realized_vol_{window}s'] = returns.rolling(window).std() * np.sqrt(window)
            df_features[f'{prefix}realized_vol_log_{window}s'] = np.log(returns.rolling(window).std() * np.sqrt(window) + 1e-8)
            
            # Price statistics
            df_features[f'{prefix}price_max_{window}s'] = df_features['mid_price'].rolling(window).max()
            df_features[f'{prefix}price_min_{window}s'] = df_features['mid_price'].rolling(window).min()
            df_features[f'{prefix}price_range_{window}s'] = (df_features[f'{prefix}price_max_{window}s'] - df_features[f'{prefix}price_min_{window}s']) / df_features['mid_price']
            
            # Volume features
            if f'{prefix}total_volume' in df_features.columns:
                df_features[f'{prefix}volume_ma_{window}s'] = df_features[f'{prefix}total_volume'].rolling(window).mean()
                df_features[f'{prefix}volume_std_{window}s'] = df_features[f'{prefix}total_volume'].rolling(window).std()
                df_features[f'{prefix}volume_momentum_{window}s'] = df_features[f'{prefix}total_volume'].pct_change(window)
            
            # Spread features
            if f'{prefix}spread' in df_features.columns:
                df_features[f'{prefix}spread_ma_{window}s'] = df_features[f'{prefix}spread'].rolling(window).mean()
                df_features[f'{prefix}spread_std_{window}s'] = df_features[f'{prefix}spread'].rolling(window).std()
                df_features[f'{prefix}spread_momentum_{window}s'] = df_features[f'{prefix}spread'].pct_change(window)
        
        # Technical indicators
        if len(df_features) > 50:
            # Moving averages and their ratios
            for ma_window in [10, 20, 50]:
                df_features[f'{prefix}sma_{ma_window}'] = df_features['mid_price'].rolling(ma_window).mean()
                df_features[f'{prefix}ema_{ma_window}'] = df_features['mid_price'].ewm(span=ma_window).mean()
                df_features[f'{prefix}price_to_sma_{ma_window}'] = df_features['mid_price'] / df_features[f'{prefix}sma_{ma_window}']
            
            # Bollinger Bands
            sma_20 = df_features['mid_price'].rolling(20).mean()
            std_20 = df_features['mid_price'].rolling(20).std()
            df_features[f'{prefix}bb_upper'] = sma_20 + (std_20 * 2)
            df_features[f'{prefix}bb_lower'] = sma_20 - (std_20 * 2)
            df_features[f'{prefix}bb_position'] = (df_features['mid_price'] - df_features[f'{prefix}bb_lower']) / (df_features[f'{prefix}bb_upper'] - df_features[f'{prefix}bb_lower'] + 1e-8)
            df_features[f'{prefix}bb_width'] = (df_features[f'{prefix}bb_upper'] - df_features[f'{prefix}bb_lower']) / df_features['mid_price']
            
            # RSI (Relative Strength Index)
            delta = df_features['mid_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            df_features[f'{prefix}rsi'] = 100 - (100 / (1 + rs))
        
        return df_features
    
    def calculate_cross_asset_features(self, eth_df):
        """Calculate cross-asset correlation and momentum features"""
        df_features = eth_df.copy()
        
        for asset_name in ['BTC', 'SOL']:
            if f"{asset_name}_train" in self.cross_assets:
                asset_df = self.cross_assets[f"{asset_name}_train"]
                
                # Merge on timestamp using backward search
                merged = pd.merge_asof(
                    df_features.sort_values('timestamp'), 
                    asset_df[['timestamp', 'mid_price']].sort_values('timestamp'),
                    on='timestamp', 
                    suffixes=('', f'_{asset_name}'),
                    direction='backward',
                    tolerance=pd.Timedelta('10s')  # Allow up to 10s tolerance
                )
                
                if f'mid_price_{asset_name}' in merged.columns:
                    # Calculate returns
                    eth_returns = merged['mid_price'].pct_change()
                    asset_returns = merged[f'mid_price_{asset_name}'].pct_change()
                    
                    # Rolling correlations at different time horizons
                    for window in [30, 60, 300]:
                        df_features[f'{asset_name}_correlation_{window}s'] = eth_returns.rolling(window).corr(asset_returns)
                    
                    # Cross-asset momentum features
                    for period in [5, 30, 60]:
                        df_features[f'{asset_name}_return_{period}s'] = merged[f'mid_price_{asset_name}'].pct_change(period)
                        df_features[f'{asset_name}_relative_return_{period}s'] = eth_returns.rolling(period).sum() - asset_returns.rolling(period).sum()
                    
                    # Cross-asset volatility features
                    asset_vol_60s = asset_returns.rolling(60).std() * np.sqrt(60)
                    eth_vol_60s = eth_returns.rolling(60).std() * np.sqrt(60)
                    df_features[f'{asset_name}_vol_ratio_60s'] = eth_vol_60s / (asset_vol_60s + 1e-8)
        
        return df_features
    
    def create_lagged_features(self, df, target_col=None):
        """Create lagged features for time series prediction"""
        df_features = df.copy()
        
        # Key features to create lags for
        lag_features = [
            'mid_price', 'spread_pct', 'volume_imbalance_1', 'realized_vol_60s',
            'return_30s', 'total_volume', 'microprice'
        ]
        
        # Include target if available (for training data)
        if target_col and target_col in df.columns:
            lag_features.append(target_col)
        
        # Different lag periods (1s to 10s)
        lags = [1, 2, 3, 5, 10]
        
        for feature in lag_features:
            if feature in df_features.columns:
                for lag in lags:
                    df_features[f'{feature}_lag_{lag}'] = df_features[feature].shift(lag)
        
        # Create rolling statistics of lagged features
        for feature in ['return_5s', 'volume_imbalance_1']:
            if feature in df_features.columns:
                df_features[f'{feature}_rolling_mean_10'] = df_features[feature].rolling(10).mean()
                df_features[f'{feature}_rolling_std_10'] = df_features[feature].rolling(10).std()
        
        return df_features
    
    def engineer_features(self):
        """Main feature engineering pipeline"""
        print("Starting feature engineering...")
        
        # Clean data first
        print("Cleaning data...")
        self.eth_train = self.clean_data(self.eth_train)
        self.eth_test = self.clean_data(self.eth_test)
        
        # Calculate orderbook features
        print("Calculating orderbook features...")
        self.eth_train = self.calculate_orderbook_features(self.eth_train)
        self.eth_test = self.calculate_orderbook_features(self.eth_test)
        
        # Calculate time series features
        print("Calculating time series features...")
        self.eth_train = self.calculate_time_series_features(self.eth_train)
        self.eth_test = self.calculate_time_series_features(self.eth_test)
        
        # Calculate cross-asset features (only for training data where we have cross-asset data)
        print("Calculating cross-asset features...")
        self.eth_train = self.calculate_cross_asset_features(self.eth_train)
        
        # Create lagged features
        print("Creating lagged features...")
        self.eth_train = self.create_lagged_features(self.eth_train, 'label')
        self.eth_test = self.create_lagged_features(self.eth_test)
        
        # Handle NaN values
        initial_train_size = len(self.eth_train)
        self.eth_train = self.eth_train.dropna()
        print(f"Removed {initial_train_size - len(self.eth_train)} rows with NaN values from training data")
        
        # For test data, use forward fill and backward fill to handle NaN
        self.eth_test = self.eth_test.fillna(method='ffill').fillna(method='bfill')
        remaining_nan = self.eth_test.isnull().sum().sum()
        if remaining_nan > 0:
            print(f"Warning: {remaining_nan} NaN values remain in test data")
            self.eth_test = self.eth_test.fillna(0)
        
        # Select feature columns
        exclude_cols = ['timestamp', 'label']
        self.feature_columns = [col for col in self.eth_train.columns if col not in exclude_cols]
        
        print(f"Created {len(self.feature_columns)} features")
        print("Feature engineering completed!")
    
    def prepare_data_for_modeling(self):
        """Prepare features and target for modeling"""
        # Ensure test data has same columns as training data
        missing_in_test = set(self.feature_columns) - set(self.eth_test.columns)
        if missing_in_test:
            print(f"Warning: {len(missing_in_test)} features missing in test data. Filling with zeros.")
            for col in missing_in_test:
                self.eth_test[col] = 0
        
        X_train = self.eth_train[self.feature_columns].copy()
        y_train = self.eth_train['label'].copy()
        X_test = self.eth_test[self.feature_columns].copy()
        
        # Handle infinite values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN with median (computed from training data)
        X_train_median = X_train.median()
        X_train = X_train.fillna(X_train_median)
        X_test = X_test.fillna(X_train_median)
        
        print(f"Final data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}")
        return X_train, y_train, X_test
    
    def time_series_cross_validation(self, X, y, n_splits=5):
        """Perform time series cross validation to estimate model performance"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        # Use Ridge regression for quick CV estimation
        model = Ridge(alpha=1.0)
        scaler = StandardScaler()
        
        print("Performing time series cross validation...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train_cv)
            X_val_scaled = scaler.transform(X_val_cv)
            
            # Train and predict
            model.fit(X_train_scaled, y_train_cv)
            y_pred = model.predict(X_val_scaled)
            
            # Calculate Pearson correlation (evaluation metric)
            corr, _ = pearsonr(y_val_cv, y_pred)
            cv_scores.append(corr)
            print(f"  Fold {fold + 1}: Correlation = {corr:.4f}")
        
        return cv_scores
    
    def train_models(self):
        """Train multiple models and create ensemble"""
        print("Preparing data for modeling...")
        X_train, y_train, X_test = self.prepare_data_for_modeling()
        
        # Perform cross validation
        cv_scores = self.time_series_cross_validation(X_train, y_train)
        print(f"CV Results - Mean: {np.mean(cv_scores):.4f}, Std: {np.std(cv_scores):.4f}")
        
        # Scale features for models that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        print("Training ensemble of models...")
        
        # Model 1: Ridge Regression (linear baseline)
        print("  Training Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        self.models['ridge'] = ridge
        
        # Model 2: Random Forest (non-linear, handles interactions)
        print("  Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # Model 3: Gradient Boosting (sequential learning)
        print("  Training Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb
        
        # Evaluate models on training data
        print("\nModel performance on training data:")
        for model_name, model in self.models.items():
            if model_name == 'ridge':
                y_pred = model.predict(X_train_scaled)
            else:
                y_pred = model.predict(X_train)
            
            corr, _ = pearsonr(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            print(f"  {model_name}: Correlation = {corr:.4f}, RMSE = {rmse:.6f}")
        
        print("Model training completed!")
        return X_test_scaled, X_test
    
    def generate_predictions(self, X_test_scaled, X_test):
        """Generate ensemble predictions"""
        print("Generating predictions...")
        
        predictions = {}
        
        # Get predictions from each model
        predictions['ridge'] = self.models['ridge'].predict(X_test_scaled)
        predictions['random_forest'] = self.models['random_forest'].predict(X_test)
        predictions['gradient_boosting'] = self.models['gradient_boosting'].predict(X_test)
        
        # Ensemble prediction with optimized weights
        # Weights based on typical performance of each model type for volatility prediction
        weights = {
            'ridge': 0.25,           # Linear baseline
            'random_forest': 0.35,   # Good for non-linear patterns
            'gradient_boosting': 0.40 # Often best for time series
        }
        
        ensemble_pred = np.zeros(len(X_test))
        for model_name, weight in weights.items():
            ensemble_pred += weight * predictions[model_name]
        
        # Apply post-processing constraints
        # Implied volatility should be positive and within reasonable bounds
        ensemble_pred = np.clip(ensemble_pred, 0.001, 1.0)
        
        print(f"Generated {len(ensemble_pred)} predictions")
        return ensemble_pred, predictions
    
    def create_submission(self, predictions):
        """Create submission file in required format"""
        submission = pd.DataFrame({
            'timestamp': range(len(predictions)),
            'labels': predictions
        })
        
        submission.to_csv("submission.csv", index=False)
        print(f"Submission file created: submission.csv")
        print(f"Submission shape: {submission.shape}")
        
        return submission
    
    def run_full_pipeline(self):
        """Execute the complete forecasting pipeline"""
        print("=" * 70)
        print("ETH IMPLIED VOLATILITY FORECASTING PIPELINE")
        print("=" * 70)
        
        try:
            # Step 1: Load data
            print("\n1. Loading data...")
            self.load_data()
            
            # Step 2: Feature engineering
            print("\n2. Feature engineering...")
            self.engineer_features()
            
            # Step 3: Train models
            print("\n3. Training models...")
            X_test_scaled, X_test = self.train_models()
            
            # Step 4: Generate predictions
            print("\n4. Generating predictions...")
            final_predictions, individual_predictions = self.generate_predictions(X_test_scaled, X_test)
            
            # Step 5: Create submission
            print("\n5. Creating submission...")
            submission = self.create_submission(final_predictions)
            
            # Summary
            print("\n" + "=" * 70)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"Total features used: {len(self.feature_columns)}")
            print(f"Predictions generated: {len(final_predictions)}")
            print(f"Prediction statistics:")
            print(f"  Mean: {np.mean(final_predictions):.6f}")
            print(f"  Std:  {np.std(final_predictions):.6f}")
            print(f"  Min:  {np.min(final_predictions):.6f}")
            print(f"  Max:  {np.max(final_predictions):.6f}")
            print("\nSubmission file 'submission.csv' is ready for upload!")
            
            return submission, final_predictions
            
        except Exception as e:
            print(f"\nError in pipeline: {str(e)}")
            raise

# Usage example and main execution
if __name__ == "__main__":
    """
    Main execution script
    
    To use this code:
    1. Ensure your data files are in the correct directories:
       - Training data: train/ETH.csv, train/BTC.csv, train/SOL.csv, etc.
       - Test data: test/ETH.csv
    2. Install required packages: pip install -r requirements.txt
    3. Run: python eth_iv_forecasting_simple.py
    """
    
    # Initialize the forecaster
    # Adjust paths as needed for your data location
    forecaster = ETHVolatilityForecaster(
        data_path_train="train/", 
        data_path_test="test/"
    )
    
    # Run the complete pipeline
    submission, predictions = forecaster.run_full_pipeline()
    
    # Optional: Display sample predictions
    print("\nSample predictions:")
    print(submission.head(10))
    print("...")
    print(submission.tail(5))