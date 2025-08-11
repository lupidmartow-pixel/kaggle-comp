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
import xgboost as xgb
import lightgbm as lgb

# Statistical imports
from scipy import stats
from scipy.stats import pearsonr
import ta  # Technical analysis library

class ETHVolatilityForecaster:
    """
    ETH Implied Volatility Forecasting Model
    
    This class implements a comprehensive approach to forecasting 10-second ahead
    implied volatility for Ethereum using high-frequency orderbook data and 
    cross-asset signals.
    """
    
    def __init__(self, data_path_train="/kaggle/input/gq-implied-volatility-forecasting/train/",
                 data_path_test="/kaggle/input/gq-implied-volatility-forecasting/test/"):
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
        
        # Load cross-asset data
        cross_asset_files = ['BTC.csv', 'SOL.csv']  # Add more as available
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
        
        # Handle missing timestamps (forward fill small gaps)
        df_clean = df_clean.set_index('timestamp').resample('1S').first().reset_index()
        
        # Forward fill missing values for small gaps (up to 10 seconds)
        for col in df_clean.columns:
            if col != 'timestamp':
                df_clean[col] = df_clean[col].fillna(method='ffill', limit=10)
        
        # Remove rows with excessive missing data
        missing_threshold = 0.5  # Remove rows with >50% missing data
        df_clean = df_clean.dropna(thresh=len(df_clean.columns) * missing_threshold)
        
        # Handle outliers using IQR method
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'label':  # Don't remove outliers from target
                Q1 = df_clean[col].quantile(0.01)
                Q3 = df_clean[col].quantile(0.99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        return df_clean
    
    def calculate_orderbook_features(self, df, prefix=""):
        """Calculate orderbook-based features"""
        df_features = df.copy()
        
        # Basic price features
        df_features[f'{prefix}spread'] = df_features['ask_price1'] - df_features['bid_price1']
        df_features[f'{prefix}spread_pct'] = df_features[f'{prefix}spread'] / df_features['mid_price']
        
        # Orderbook imbalance features
        df_features[f'{prefix}volume_imbalance_1'] = (df_features['bid_volume1'] - df_features['ask_volume1']) / (df_features['bid_volume1'] + df_features['ask_volume1'] + 1e-8)
        
        # Total volume features
        bid_volumes = [f'bid_volume{i}' for i in range(1, 6)]
        ask_volumes = [f'ask_volume{i}' for i in range(1, 6)]
        
        df_features[f'{prefix}total_bid_volume'] = df_features[bid_volumes].sum(axis=1)
        df_features[f'{prefix}total_ask_volume'] = df_features[ask_volumes].sum(axis=1)
        df_features[f'{prefix}total_volume'] = df_features[f'{prefix}total_bid_volume'] + df_features[f'{prefix}total_ask_volume']
        
        # Volume-weighted prices
        bid_prices = [f'bid_price{i}' for i in range(1, 6)]
        ask_prices = [f'ask_price{i}' for i in range(1, 6)]
        
        df_features[f'{prefix}vwap_bid'] = (df_features[bid_prices] * df_features[bid_volumes]).sum(axis=1) / (df_features[bid_volumes].sum(axis=1) + 1e-8)
        df_features[f'{prefix}vwap_ask'] = (df_features[ask_prices] * df_features[ask_volumes]).sum(axis=1) / (df_features[ask_volumes].sum(axis=1) + 1e-8)
        
        # Depth features
        df_features[f'{prefix}depth_5_levels'] = df_features[f'{prefix}total_volume']
        df_features[f'{prefix}price_impact_buy'] = (df_features['ask_price5'] - df_features['ask_price1']) / df_features['ask_price1']
        df_features[f'{prefix}price_impact_sell'] = (df_features['bid_price1'] - df_features['bid_price5']) / df_features['bid_price1']
        
        return df_features
    
    def calculate_time_series_features(self, df, prefix=""):
        """Calculate time-series based features"""
        df_features = df.copy()
        
        # Price-based features
        windows = [5, 10, 30, 60, 300]  # 5s, 10s, 30s, 1min, 5min
        
        for window in windows:
            # Returns
            df_features[f'{prefix}return_{window}s'] = df_features['mid_price'].pct_change(window)
            
            # Realized volatility
            returns = df_features['mid_price'].pct_change()
            df_features[f'{prefix}realized_vol_{window}s'] = returns.rolling(window).std() * np.sqrt(window)
            
            # Price momentum
            df_features[f'{prefix}momentum_{window}s'] = (df_features['mid_price'] / df_features['mid_price'].shift(window) - 1)
            
            # Volume momentum
            df_features[f'{prefix}volume_momentum_{window}s'] = df_features[f'{prefix}total_volume'].pct_change(window)
            
            # Spread momentum
            df_features[f'{prefix}spread_momentum_{window}s'] = df_features[f'{prefix}spread'].pct_change(window)
        
        # Technical indicators
        if len(df_features) > 50:  # Ensure enough data for indicators
            # Moving averages
            df_features[f'{prefix}sma_20'] = df_features['mid_price'].rolling(20).mean()
            df_features[f'{prefix}ema_20'] = df_features['mid_price'].ewm(span=20).mean()
            
            # Bollinger Bands
            sma_20 = df_features['mid_price'].rolling(20).mean()
            std_20 = df_features['mid_price'].rolling(20).std()
            df_features[f'{prefix}bb_upper'] = sma_20 + (std_20 * 2)
            df_features[f'{prefix}bb_lower'] = sma_20 - (std_20 * 2)
            df_features[f'{prefix}bb_position'] = (df_features['mid_price'] - df_features[f'{prefix}bb_lower']) / (df_features[f'{prefix}bb_upper'] - df_features[f'{prefix}bb_lower'])
            
            # RSI
            delta = df_features['mid_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_features[f'{prefix}rsi'] = 100 - (100 / (1 + rs))
        
        return df_features
    
    def calculate_cross_asset_features(self, eth_df):
        """Calculate cross-asset correlation and momentum features"""
        df_features = eth_df.copy()
        
        for asset_name in ['BTC', 'SOL']:
            if f"{asset_name}_train" in self.cross_assets:
                asset_df = self.cross_assets[f"{asset_name}_train"]
                
                # Merge on timestamp
                merged = pd.merge_asof(df_features.sort_values('timestamp'), 
                                     asset_df[['timestamp', 'mid_price']].sort_values('timestamp'),
                                     on='timestamp', 
                                     suffixes=('', f'_{asset_name}'),
                                     direction='backward')
                
                if f'mid_price_{asset_name}' in merged.columns:
                    # Cross-asset returns correlation
                    eth_returns = merged['mid_price'].pct_change()
                    asset_returns = merged[f'mid_price_{asset_name}'].pct_change()
                    
                    # Rolling correlation
                    df_features[f'{asset_name}_correlation_60s'] = eth_returns.rolling(60).corr(asset_returns)
                    
                    # Cross-asset momentum
                    df_features[f'{asset_name}_return_5s'] = merged[f'mid_price_{asset_name}'].pct_change(5)
                    df_features[f'{asset_name}_return_30s'] = merged[f'mid_price_{asset_name}'].pct_change(30)
                    
                    # Relative performance
                    df_features[f'{asset_name}_relative_performance'] = eth_returns - asset_returns
        
        return df_features
    
    def create_lagged_features(self, df, target_col=None):
        """Create lagged features for time series prediction"""
        df_features = df.copy()
        
        # Lag features for key variables
        lag_features = ['mid_price', 'spread_pct', 'volume_imbalance_1', 'realized_vol_60s']
        if target_col and target_col in df.columns:
            lag_features.append(target_col)
        
        lags = [1, 2, 3, 5, 10]  # 1s, 2s, 3s, 5s, 10s lags
        
        for feature in lag_features:
            if feature in df_features.columns:
                for lag in lags:
                    df_features[f'{feature}_lag_{lag}'] = df_features[feature].shift(lag)
        
        return df_features
    
    def engineer_features(self):
        """Main feature engineering pipeline"""
        print("Starting feature engineering...")
        
        # Clean data
        self.eth_train = self.clean_data(self.eth_train)
        self.eth_test = self.clean_data(self.eth_test)
        
        # Calculate orderbook features
        self.eth_train = self.calculate_orderbook_features(self.eth_train)
        self.eth_test = self.calculate_orderbook_features(self.eth_test)
        
        # Calculate time series features
        self.eth_train = self.calculate_time_series_features(self.eth_train)
        self.eth_test = self.calculate_time_series_features(self.eth_test)
        
        # Calculate cross-asset features
        self.eth_train = self.calculate_cross_asset_features(self.eth_train)
        # Note: For test data, we'd need cross-asset test data to calculate these features
        
        # Create lagged features
        self.eth_train = self.create_lagged_features(self.eth_train, 'label')
        self.eth_test = self.create_lagged_features(self.eth_test)
        
        # Remove rows with NaN values created by feature engineering
        initial_train_size = len(self.eth_train)
        self.eth_train = self.eth_train.dropna()
        print(f"Removed {initial_train_size - len(self.eth_train)} rows with NaN values from training data")
        
        # Fill NaN in test data with forward fill and then backward fill
        self.eth_test = self.eth_test.fillna(method='ffill').fillna(method='bfill')
        
        # Select feature columns (exclude timestamp and label)
        exclude_cols = ['timestamp', 'label']
        self.feature_columns = [col for col in self.eth_train.columns if col not in exclude_cols]
        
        print(f"Created {len(self.feature_columns)} features")
        print("Feature engineering completed!")
    
    def prepare_data_for_modeling(self):
        """Prepare features and target for modeling"""
        X_train = self.eth_train[self.feature_columns].copy()
        y_train = self.eth_train['label'].copy()
        X_test = self.eth_test[self.feature_columns].copy()
        
        # Handle any remaining infinite values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())  # Use training median for test
        
        return X_train, y_train, X_test
    
    def time_series_cross_validation(self, X, y, n_splits=5):
        """Perform time series cross validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        # Use a simple model for CV to save time
        model = Ridge(alpha=1.0)
        scaler = StandardScaler()
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train_cv)
            X_val_scaled = scaler.transform(X_val_cv)
            
            # Train and predict
            model.fit(X_train_scaled, y_train_cv)
            y_pred = model.predict(X_val_scaled)
            
            # Calculate correlation
            corr, _ = pearsonr(y_val_cv, y_pred)
            cv_scores.append(corr)
        
        return cv_scores
    
    def train_models(self):
        """Train multiple models and ensemble them"""
        print("Preparing data for modeling...")
        X_train, y_train, X_test = self.prepare_data_for_modeling()
        
        print("Performing time series cross validation...")
        cv_scores = self.time_series_cross_validation(X_train, y_train)
        print(f"CV Correlation scores: {cv_scores}")
        print(f"Mean CV Correlation: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        print("Training models...")
        
        # Model 1: Ridge Regression
        print("Training Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        self.models['ridge'] = ridge
        
        # Model 2: Random Forest
        print("Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)  # RF doesn't need scaling
        self.models['random_forest'] = rf
        
        # Model 3: XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # Model 4: LightGBM
        print("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_model
        
        # Evaluate models on training data
        print("\nModel evaluation on training data:")
        for model_name, model in self.models.items():
            if model_name == 'ridge':
                y_pred = model.predict(X_train_scaled)
            else:
                y_pred = model.predict(X_train)
            
            corr, _ = pearsonr(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            print(f"{model_name}: Correlation = {corr:.4f}, RMSE = {rmse:.6f}")
        
        print("Model training completed!")
        
        return X_test_scaled, X_test
    
    def generate_predictions(self, X_test_scaled, X_test):
        """Generate ensemble predictions"""
        print("Generating predictions...")
        
        predictions = {}
        
        # Get predictions from each model
        predictions['ridge'] = self.models['ridge'].predict(X_test_scaled)
        predictions['random_forest'] = self.models['random_forest'].predict(X_test)
        predictions['xgboost'] = self.models['xgboost'].predict(X_test)
        predictions['lightgbm'] = self.models['lightgbm'].predict(X_test)
        
        # Ensemble prediction (weighted average)
        weights = {
            'ridge': 0.2,
            'random_forest': 0.2,
            'xgboost': 0.3,
            'lightgbm': 0.3
        }
        
        ensemble_pred = np.zeros(len(X_test))
        for model_name, weight in weights.items():
            ensemble_pred += weight * predictions[model_name]
        
        return ensemble_pred, predictions
    
    def create_submission(self, predictions):
        """Create submission file"""
        submission = pd.DataFrame({
            'timestamp': range(len(predictions)),
            'labels': predictions
        })
        
        submission.to_csv("/workspace/submission.csv", index=False)
        print("Submission file created: submission.csv")
        
        return submission
    
    def run_full_pipeline(self):
        """Run the complete forecasting pipeline"""
        print("Starting ETH Implied Volatility Forecasting Pipeline...")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Engineer features
        self.engineer_features()
        
        # Train models
        X_test_scaled, X_test = self.train_models()
        
        # Generate predictions
        final_predictions, individual_predictions = self.generate_predictions(X_test_scaled, X_test)
        
        # Create submission
        submission = self.create_submission(final_predictions)
        
        print("=" * 60)
        print("Pipeline completed successfully!")
        print(f"Final predictions shape: {final_predictions.shape}")
        print(f"Prediction statistics:")
        print(f"  Mean: {np.mean(final_predictions):.6f}")
        print(f"  Std:  {np.std(final_predictions):.6f}")
        print(f"  Min:  {np.min(final_predictions):.6f}")
        print(f"  Max:  {np.max(final_predictions):.6f}")
        
        return submission, final_predictions

# Main execution
if __name__ == "__main__":
    # Initialize the forecaster
    forecaster = ETHVolatilityForecaster()
    
    # Run the complete pipeline
    submission, predictions = forecaster.run_full_pipeline()
    
    # Display submission preview
    print("\nSubmission preview:")
    print(submission.head(10))