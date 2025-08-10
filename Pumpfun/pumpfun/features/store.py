"""
Feature store for managing feature computation and storage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pickle
import json
from dataclasses import dataclass, asdict
import time

from ..config import config
from .engineers import (
    ChainFlowEngineer,
    WalletQualityEngineer,
    SocialEngineer,
    ContentEngineer,
    ImageEngineer,
    RegimeEngineer
)


@dataclass
class FeatureMetadata:
    """Metadata for a feature set."""
    
    token_id: str
    timestamp: datetime
    feature_count: int
    feature_names: List[str]
    feature_families: List[str]
    computation_time: float
    data_sources: List[str]
    version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureMetadata':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class FeatureStore:
    """Manages feature computation, storage, and retrieval."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or config.data.features_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature engineers
        self.engineers = {
            'chain_flow': ChainFlowEngineer(),
            'wallet_quality': WalletQualityEngineer(),
            'social': SocialEngineer(),
            'content': ContentEngineer(),
            'image': ImageEngineer(),
            'regime': RegimeEngineer()
        }
        
        # Feature metadata storage
        self.metadata_file = self.base_dir / "feature_metadata.json"
        self.feature_metadata = self._load_metadata()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def _load_metadata(self) -> Dict[str, List[FeatureMetadata]]:
        """Load feature metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata_data = json.load(f)
                
                metadata = {}
                for token_id, metadata_list in metadata_data.items():
                    metadata[token_id] = [FeatureMetadata.from_dict(m) for m in metadata_list]
                return metadata
            except Exception as e:
                self.logger.error(f"Failed to load feature metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save feature metadata to disk."""
        try:
            metadata_data = {}
            for token_id, metadata_list in self.feature_metadata.items():
                metadata_data[token_id] = [m.to_dict() for m in metadata_list]
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_data, f, default=str, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save feature metadata: {e}")
    
    def compute_features(
        self,
        token_data: Dict[str, pd.DataFrame],
        token_id: str,
        timestamp: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, FeatureMetadata]:
        """Compute features for a token from raw data."""
        if timestamp is None:
            timestamp = datetime.now()
        
        start_time = time.time()
        
        try:
            # Compute features for each family
            all_features = {}
            feature_families = []
            data_sources = []
            
            for family_name, engineer in self.engineers.items():
                if family_name in config.feature.feature_families:
                    try:
                        self.logger.info(f"Computing {family_name} features for {token_id}")
                        
                        # Get relevant data for this family
                        family_data = self._get_family_data(family_name, token_data)
                        
                        if not family_data.empty:
                            features = engineer.compute_features(family_data, token_id, timestamp)
                            if not features.empty:
                                all_features[f"{family_name}_"] = features
                                feature_families.append(family_name)
                                data_sources.extend(engineer.get_data_sources())
                                
                                self.logger.info(f"Computed {len(features.columns)} {family_name} features")
                        else:
                            self.logger.warning(f"No data available for {family_name} features")
                    
                    except Exception as e:
                        self.logger.error(f"Failed to compute {family_name} features: {e}")
                        continue
            
            # Combine all features
            if not all_features:
                raise ValueError("No features were computed successfully")
            
            # Concatenate features horizontally
            combined_features = pd.concat(all_features.values(), axis=1)
            
            # Add token identifier and timestamp
            combined_features['token_id'] = token_id
            combined_features['timestamp'] = timestamp
            
            # Ensure no duplicate columns
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            
            # Create metadata
            computation_time = time.time() - start_time
            metadata = FeatureMetadata(
                token_id=token_id,
                timestamp=timestamp,
                feature_count=len(combined_features.columns) - 2,  # Exclude token_id and timestamp
                feature_names=list(combined_features.columns),
                feature_families=feature_families,
                computation_time=computation_time,
                data_sources=list(set(data_sources)),
                version=config.model.model_version_format.format(
                    date=timestamp.strftime("%Y.%m.%d"),
                    version="01"
                )
            )
            
            self.logger.info(f"Computed {metadata.feature_count} features for {token_id} in {computation_time:.2f}s")
            
            return combined_features, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to compute features for {token_id}: {e}")
            raise
    
    def _get_family_data(self, family_name: str, token_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Get relevant data for a specific feature family."""
        if family_name == 'chain_flow':
            return token_data.get('solana', pd.DataFrame())
        elif family_name == 'wallet_quality':
            return token_data.get('solana', pd.DataFrame())
        elif family_name == 'social':
            return token_data.get('social', pd.DataFrame())
        elif family_name == 'content':
            return token_data.get('pumpfun', pd.DataFrame())
        elif family_name == 'image':
            return token_data.get('pumpfun', pd.DataFrame())
        elif family_name == 'regime':
            # Regime features can use data from multiple sources
            regime_data = {}
            for source, data in token_data.items():
                if not data.empty and 'timestamp' in data.columns:
                    regime_data[source] = data
            return regime_data
        else:
            return pd.DataFrame()
    
    def store_features(
        self,
        token_id: str,
        features: pd.DataFrame,
        metadata: FeatureMetadata
    ) -> Path:
        """Store computed features and metadata."""
        try:
            # Create token-specific directory
            token_dir = self.base_dir / token_id
            token_dir.mkdir(parents=True, exist_ok=True)
            
            # Store features
            timestamp_str = metadata.timestamp.strftime("%Y%m%d_%H%M%S")
            features_file = token_dir / f"features_{timestamp_str}.parquet"
            features.to_parquet(features_file, engine='pyarrow', compression='snappy')
            
            # Store metadata
            metadata_file = token_dir / f"metadata_{timestamp_str}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, default=str, indent=2)
            
            # Update metadata registry
            if token_id not in self.feature_metadata:
                self.feature_metadata[token_id] = []
            self.feature_metadata[token_id].append(metadata)
            self._save_metadata()
            
            self.logger.info(f"Stored features for {token_id} at {features_file}")
            return features_file
            
        except Exception as e:
            self.logger.error(f"Failed to store features for {token_id}: {e}")
            raise
    
    def load_features(
        self,
        token_id: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Load features for a token."""
        if token_id not in self.feature_metadata:
            return None
        
        metadata_list = self.feature_metadata[token_id]
        if not metadata_list:
            return None
        
        if timestamp is None:
            # Load latest features
            metadata = max(metadata_list, key=lambda m: m.timestamp)
        else:
            # Load features closest to timestamp
            metadata = min(metadata_list, key=lambda m: abs((m.timestamp - timestamp).total_seconds()))
        
        # Load features from file
        token_dir = self.base_dir / token_id
        timestamp_str = metadata.timestamp.strftime("%Y%m%d_%H%M%S")
        features_file = token_dir / f"features_{timestamp_str}.parquet"
        
        if not features_file.exists():
            self.logger.warning(f"Features file not found: {features_file}")
            return None
        
        try:
            features = pd.read_parquet(features_file)
            self.logger.info(f"Loaded {len(features.columns)} features for {token_id}")
            return features
        except Exception as e:
            self.logger.error(f"Failed to load features for {token_id}: {e}")
            return None
    
    def get_feature_summary(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Get feature summary for a token."""
        if token_id not in self.feature_metadata:
            return None
        
        metadata_list = self.feature_metadata[token_id]
        if not metadata_list:
            return None
        
        latest_metadata = max(metadata_list, key=lambda m: m.timestamp)
        
        return {
            'token_id': token_id,
            'latest_features': latest_metadata.to_dict(),
            'total_feature_sets': len(metadata_list),
            'feature_history': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'feature_count': m.feature_count,
                    'families': m.feature_families
                }
                for m in metadata_list
            ]
        }
    
    def get_all_tokens(self) -> List[str]:
        """Get list of all tokens with features."""
        return list(self.feature_metadata.keys())
    
    def cleanup_old_features(self, max_age_hours: int = 168):
        """Clean up old feature files."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for token_id in list(self.feature_metadata.keys()):
            metadata_list = self.feature_metadata[token_id]
            metadata_to_remove = [m for m in metadata_list if m.timestamp < cutoff_time]
            
            for metadata in metadata_to_remove:
                try:
                    # Remove feature file
                    token_dir = self.base_dir / token_id
                    timestamp_str = metadata.timestamp.strftime("%Y%m%d_%H%M%S")
                    features_file = token_dir / f"features_{timestamp_str}.parquet"
                    metadata_file = token_dir / f"metadata_{timestamp_str}.json"
                    
                    if features_file.exists():
                        features_file.unlink()
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    # Remove from metadata
                    metadata_list.remove(metadata)
                    
                    self.logger.info(f"Removed old features for {token_id} from {metadata.timestamp}")
                except Exception as e:
                    self.logger.error(f"Failed to remove old features for {token_id}: {e}")
            
            # Remove token if no metadata remains
            if not metadata_list:
                del self.feature_metadata[token_id]
        
        # Save updated metadata
        self._save_metadata()
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature quality and consistency."""
        validation_report = {
            'timestamp': datetime.now(),
            'status': 'valid',
            'issues': [],
            'statistics': {}
        }
        
        try:
            # Check for missing values
            missing_counts = features.isnull().sum()
            high_missing_features = missing_counts[missing_counts > len(features) * 0.5]
            
            if not high_missing_features.empty:
                validation_report['status'] = 'degraded'
                validation_report['issues'].append(f"High missing values in features: {list(high_missing_features.index)}")
            
            # Check for infinite values
            inf_counts = np.isinf(features.select_dtypes(include=[np.number])).sum()
            if inf_counts.sum() > 0:
                validation_report['status'] = 'degraded'
                validation_report['issues'].append(f"Infinite values found in {inf_counts.sum()} cells")
            
            # Check for constant features
            constant_features = []
            for col in features.columns:
                if features[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                validation_report['issues'].append(f"Constant features: {constant_features}")
            
            # Generate statistics
            validation_report['statistics'] = {
                'total_features': len(features.columns),
                'total_samples': len(features),
                'missing_value_percentage': (features.isnull().sum().sum() / (len(features) * len(features.columns))) * 100,
                'constant_features': len(constant_features),
                'numeric_features': len(features.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(features.select_dtypes(include=['object', 'category']).columns)
            }
            
        except Exception as e:
            validation_report['status'] = 'error'
            validation_report['issues'].append(f"Validation error: {e}")
        
        return validation_report


# Convenience function for quick feature computation
def compute_token_features(
    token_data: Dict[str, pd.DataFrame],
    token_id: str,
    timestamp: Optional[datetime] = None
) -> Tuple[pd.DataFrame, FeatureMetadata]:
    """Quick feature computation using default feature store."""
    feature_store = FeatureStore()
    return feature_store.compute_features(token_data, token_id, timestamp)
