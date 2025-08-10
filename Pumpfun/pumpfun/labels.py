"""
Label generation for the Pump.fun token ranking system.

This module implements label building for various prediction targets:
- Return predictions (1h, 6h, 24h)
- Liquidity survival
- Sustained activity
- Downside control
- Market regime conditioning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from loguru import logger


class LabelBuilder:
    """Builds training labels from historical token data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.label_encoders = {}
        self.label_metadata = {}
        
    def build_return_labels(self, data: pd.DataFrame, 
                          price_col: str = 'price',
                          timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Build return-based prediction labels."""
        logger.info("Building return-based labels")
        
        # Sort by timestamp
        data = data.sort_values(timestamp_col).reset_index(drop=True)
        
        labels = pd.DataFrame()
        labels[timestamp_col] = data[timestamp_col]
        labels['token_id'] = data.get('token_id', range(len(data)))
        
        # Calculate returns for different horizons
        for horizon in self.config.get('return_horizons', [1, 6, 24]):
            # Convert hours to minutes for intraday data
            horizon_minutes = horizon * 60
            
            # Calculate forward returns
            future_prices = data[price_col].shift(-horizon_minutes)
            returns = (future_prices - data[price_col]) / data[price_col]
            
            # Create binary labels based on thresholds
            for threshold in self.config.get('return_thresholds', [0.1, 0.25, 0.5, 1.0]):
                label_name = f'return_{horizon}h_{int(threshold*100)}pct'
                labels[label_name] = (returns > threshold).astype(int)
                
            # Store continuous returns
            labels[f'return_{horizon}h_continuous'] = returns
            
        # Handle missing values (future returns that don't exist)
        return_cols = [col for col in labels.columns if col.startswith('return_')]
        labels[return_cols] = labels[return_cols].fillna(0)
        
        logger.info(f"Built {len(return_cols)} return labels")
        return labels
        
    def build_liquidity_labels(self, data: pd.DataFrame,
                             liquidity_col: str = 'liquidity_usd',
                             timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Build liquidity survival labels."""
        logger.info("Building liquidity survival labels")
        
        data = data.sort_values(timestamp_col).reset_index(drop=True)
        
        labels = pd.DataFrame()
        labels[timestamp_col] = data[timestamp_col]
        labels['token_id'] = data.get('token_id', range(len(data)))
        
        threshold = self.config.get('liquidity_survival_threshold', 1000)
        horizon = self.config.get('liquidity_survival_horizon', 6)
        horizon_minutes = horizon * 60
        
        # Check if liquidity drops below threshold within horizon
        future_liquidity = data[liquidity_col].shift(-horizon_minutes)
        liquidity_survives = (future_liquidity >= threshold).astype(int)
        
        labels['liquidity_survives'] = liquidity_survives
        
        # Time to liquidity death (if it happens)
        labels['time_to_liquidity_death'] = np.nan
        
        for i in range(len(data)):
            if data[liquidity_col].iloc[i] >= threshold:
                # Find when liquidity drops below threshold
                for j in range(i + 1, min(i + horizon_minutes + 1, len(data))):
                    if data[liquidity_col].iloc[j] < threshold:
                        labels.loc[i, 'time_to_liquidity_death'] = j - i
                        break
                        
        # Fill missing values
        labels['liquidity_survives'] = labels['liquidity_survives'].fillna(0)
        labels['time_to_liquidity_death'] = labels['time_to_liquidity_death'].fillna(horizon_minutes)
        
        logger.info("Built liquidity survival labels")
        return labels
        
    def build_activity_labels(self, data: pd.DataFrame,
                            volume_col: str = 'volume_24h',
                            tx_count_col: str = 'transaction_count',
                            timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Build sustained activity labels."""
        logger.info("Building sustained activity labels")
        
        data = data.sort_values(timestamp_col).reset_index(drop=True)
        
        labels = pd.DataFrame()
        labels[timestamp_col] = data[timestamp_col]
        labels['token_id'] = data.get('token_id', range(len(data)))
        
        percentile = self.config.get('activity_percentile', 0.75)
        duration = self.config.get('activity_duration_hours', 6)
        duration_minutes = duration * 60
        
        # Calculate activity thresholds based on historical percentiles
        volume_threshold = data[volume_col].quantile(percentile)
        tx_threshold = data[tx_count_col].quantile(percentile)
        
        # Check if activity sustains above threshold for duration
        labels['sustained_volume'] = 0
        labels['sustained_transactions'] = 0
        
        for i in range(len(data)):
            if i + duration_minutes < len(data):
                # Check volume sustainability
                future_volumes = data[volume_col].iloc[i:i+duration_minutes]
                if (future_volumes >= volume_threshold).all():
                    labels.loc[i, 'sustained_volume'] = 1
                    
                # Check transaction sustainability
                future_txs = data[tx_count_col].iloc[i:i+duration_minutes]
                if (future_txs >= tx_threshold).all():
                    labels.loc[i, 'sustained_transactions'] = 1
                    
        # Combined sustained activity
        labels['sustained_activity'] = (
            (labels['sustained_volume'] == 1) & 
            (labels['sustained_transactions'] == 1)
        ).astype(int)
        
        logger.info("Built sustained activity labels")
        return labels
        
    def build_drawdown_labels(self, data: pd.DataFrame,
                            price_col: str = 'price',
                            timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Build downside control labels."""
        logger.info("Building downside control labels")
        
        data = data.sort_values(timestamp_col).reset_index(drop=True)
        
        labels = pd.DataFrame()
        labels[timestamp_col] = data[timestamp_col]
        labels['token_id'] = data.get('token_id', range(len(data)))
        
        max_drawdown = self.config.get('max_drawdown_threshold', 0.3)
        horizon = self.config.get('drawdown_horizon', 6)
        horizon_minutes = horizon * 60
        
        # Calculate maximum drawdown within horizon
        labels['max_drawdown'] = np.nan
        labels['downside_controlled'] = 1  # Default to controlled
        
        for i in range(len(data)):
            if i + horizon_minutes < len(data):
                future_prices = data[price_col].iloc[i:i+horizon_minutes]
                peak_price = data[price_col].iloc[i]
                
                # Calculate running maximum and drawdown
                running_max = np.maximum.accumulate(future_prices)
                drawdowns = (running_max - future_prices) / running_max
                max_dd = np.max(drawdowns)
                
                labels.loc[i, 'max_drawdown'] = max_dd
                
                # Check if downside is controlled
                if max_dd > max_drawdown:
                    labels.loc[i, 'downside_controlled'] = 0
                    
        # Fill missing values
        labels['max_drawdown'] = labels['max_drawdown'].fillna(0)
        labels['downside_controlled'] = labels['downside_controlled'].fillna(1)
        
        logger.info("Built downside control labels")
        return labels
        
    def build_regime_labels(self, data: pd.DataFrame,
                          volatility_col: str = 'volatility_24h',
                          volume_col: str = 'volume_24h',
                          timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Build market regime labels."""
        logger.info("Building market regime labels")
        
        data = data.sort_values(timestamp_col).reset_index(drop=True)
        
        labels = pd.DataFrame()
        labels[timestamp_col] = data[timestamp_col]
        labels['token_id'] = data.get('token_id', range(len(data)))
        
        # Volatility regime buckets
        vol_buckets = self.config.get('regime_volatility_buckets', [0.1, 0.25, 0.5, 1.0])
        
        # Create volatility regime labels
        labels['volatility_regime'] = pd.cut(
            data[volatility_col], 
            bins=[0] + vol_buckets + [np.inf],
            labels=['low', 'medium_low', 'medium_high', 'high', 'extreme'],
            include_lowest=True
        ).astype(str)
        
        # Volume regime (relative to historical)
        volume_ma = data[volume_col].rolling(window=24*7).mean()  # 7-day MA
        volume_std = data[volume_col].rolling(window=24*7).std()
        
        labels['volume_regime'] = pd.cut(
            (data[volume_col] - volume_ma) / volume_std,
            bins=[-np.inf, -1, 0, 1, np.inf],
            labels=['very_low', 'low', 'normal', 'high'],
            include_lowest=True
        ).astype(str)
        
        # Combined regime
        labels['market_regime'] = labels['volatility_regime'] + '_' + labels['volume_regime']
        
        # Encode categorical labels
        for col in ['volatility_regime', 'volume_regime', 'market_regime']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                # Fit on all unique values including 'unknown'
                unique_values = labels[col].fillna('unknown').unique()
                self.label_encoders[col].fit(unique_values)
                labels[f'{col}_encoded'] = self.label_encoders[col].transform(labels[col].fillna('unknown'))
            else:
                # Handle new categories by refitting the encoder
                current_values = labels[col].fillna('unknown').unique()
                if not all(val in self.label_encoders[col].classes_ for val in current_values):
                    # Refit with new categories
                    all_values = list(self.label_encoders[col].classes_) + [val for val in current_values if val not in self.label_encoders[col].classes_]
                    self.label_encoders[col].fit(all_values)
                labels[f'{col}_encoded'] = self.label_encoders[col].transform(labels[col].fillna('unknown'))
                
        logger.info("Built market regime labels")
        return labels
        
    def build_all_labels(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Build all label types and combine them."""
        logger.info("Building comprehensive label set")
        
        # Build individual label sets
        return_labels = self.build_return_labels(data, **kwargs)
        liquidity_labels = self.build_liquidity_labels(data, **kwargs)
        activity_labels = self.build_activity_labels(data, **kwargs)
        drawdown_labels = self.build_drawdown_labels(data, **kwargs)
        regime_labels = self.build_regime_labels(data, **kwargs)
        
        # Merge all labels
        all_labels = return_labels.copy()
        
        for label_df in [liquidity_labels, activity_labels, drawdown_labels, regime_labels]:
            # Drop duplicate columns
            cols_to_add = [col for col in label_df.columns if col not in all_labels.columns]
            all_labels = pd.concat([all_labels, label_df[cols_to_add]], axis=1)
            
        # Store label metadata
        self.label_metadata = {
            'n_labels': len(all_labels.columns) - 2,  # Exclude timestamp and token_id
            'label_types': {
                'return': [col for col in all_labels.columns if col.startswith('return_')],
                'liquidity': [col for col in all_labels.columns if col.startswith('liquidity_')],
                'activity': [col for col in all_labels.columns if col.startswith('sustained_')],
                'drawdown': [col for col in all_labels.columns if col.startswith('max_drawdown') or col.startswith('downside_')],
                'regime': [col for col in all_labels.columns if col.startswith('volatility_') or col.startswith('volume_') or col.startswith('market_')]
            },
            'build_date': datetime.now().isoformat()
        }
        
        logger.info(f"Built {self.label_metadata['n_labels']} total labels")
        return all_labels
        
    def get_label_summary(self) -> Dict[str, Any]:
        """Get summary statistics of built labels."""
        if not self.label_metadata:
            return {}
            
        return self.label_metadata
        
    def save_labels(self, labels: pd.DataFrame, save_path: Path) -> None:
        """Save labels to disk."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save labels
        labels.to_parquet(save_path, index=False)
        
        # Save metadata
        metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.label_metadata, f, indent=2)
            
        logger.info(f"Labels saved to {save_path}")
        
    def load_labels(self, load_path: Path) -> pd.DataFrame:
        """Load labels from disk."""
        labels = pd.read_parquet(load_path)
        
        # Try to load metadata
        metadata_path = load_path.parent / f"{load_path.stem}_metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                self.label_metadata = json.load(f)
                
        logger.info(f"Labels loaded from {load_path}")
        return labels
        
    def validate_labels(self, labels: pd.DataFrame) -> Dict[str, Any]:
        """Validate label quality and consistency."""
        validation_results = {}
        
        # Check for missing values
        missing_counts = labels.isnull().sum()
        validation_results['missing_values'] = missing_counts.to_dict()
        
        # Check label distributions
        binary_cols = []
        continuous_cols = []
        
        for col in labels.columns:
            if col in ['timestamp', 'token_id']:
                continue
                
            unique_vals = labels[col].nunique()
            if unique_vals <= 2:
                binary_cols.append(col)
                validation_results[f'{col}_distribution'] = labels[col].value_counts().to_dict()
            else:
                continuous_cols.append(col)
                validation_results[f'{col}_stats'] = {
                    'mean': float(labels[col].mean()),
                    'std': float(labels[col].std()),
                    'min': float(labels[col].min()),
                    'max': float(labels[col].max())
                }
                
        validation_results['binary_labels'] = binary_cols
        validation_results['continuous_labels'] = continuous_cols
        
        # Check for class imbalance in binary labels
        for col in binary_cols:
            if col in labels.columns:
                counts = labels[col].value_counts()
                if len(counts) == 2:
                    ratio = counts.iloc[0] / counts.iloc[1]
                    validation_results[f'{col}_class_ratio'] = float(ratio)
                    
        logger.info("Label validation completed")
        return validation_results
