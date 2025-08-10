#!/usr/bin/env python3
"""
Test script for the Pump.fun token ranking system.

This script tests the core components to ensure they're working correctly.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the pumpfun package to the path
sys.path.insert(0, str(Path(__file__).parent))

from pumpfun.config import config
from pumpfun.data.connector import DataConnector
from pumpfun.data.storage import DataStorage
from pumpfun.features.engineers import FeatureEngineer
from pumpfun.features.store import FeatureStore
from pumpfun.models import ModelRegistry, TokenRankingModel
from pumpfun.labels import LabelBuilder
from pumpfun.evaluation import ModelEvaluator


def test_config():
    """Test configuration loading."""
    print("🔧 Testing configuration...")
    try:
        print(f"  - Data directory: {config.data.data_dir}")
        print(f"  - Feature families: {config.feature.feature_families}")
        print(f"  - Model families: {config.model.model_families}")
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_data_connector():
    """Test data connector initialization."""
    print("\n📡 Testing data connector...")
    try:
        connector = DataConnector()
        print(f"  - Available sources: {list(connector.sources.keys())}")
        print("✅ Data connector initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Data connector error: {e}")
        return False


def test_data_storage():
    """Test data storage initialization."""
    print("\n💾 Testing data storage...")
    try:
        storage = DataStorage(config.model_dump())
        print(f"  - Storage directory: {storage.data_dir}")
        print("✅ Data storage initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Data storage error: {e}")
        return False


def test_feature_engineers():
    """Test feature engineers."""
    print("\n⚙️ Testing feature engineers...")
    try:
        engineer = FeatureEngineer(config)
        print(f"  - Available engineers: {list(engineer.engineers.keys())}")
        print(f"  - Total feature names: {len(engineer.get_feature_names())}")
        print("✅ Feature engineers initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Feature engineers error: {e}")
        return False


def test_feature_store():
    """Test feature store."""
    print("\n🏪 Testing feature store...")
    try:
        store = FeatureStore()  # Use default config
        print(f"  - Feature store initialized")
        print("✅ Feature store initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Feature store error: {e}")
        return False


def test_model_registry():
    """Test model registry."""
    print("\n📚 Testing model registry...")
    try:
        # ModelRegistry expects a dictionary with .get() method
        registry = ModelRegistry(config.model_dump())
        print(f"  - Model registry initialized")
        print("✅ Model registry initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Model registry error: {e}")
        return False


def test_token_ranking_model():
    """Test token ranking model."""
    print("\n🤖 Testing token ranking model...")
    try:
        model = TokenRankingModel(config)
        print(f"  - Token ranking model initialized")
        print("✅ Token ranking model initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Token ranking model error: {e}")
        return False


def test_label_builder():
    """Test label builder."""
    print("\n🏷️ Testing label builder...")
    try:
        # LabelBuilder expects a dictionary with .get() method
        builder = LabelBuilder(config.model_dump())
        print(f"  - Label builder initialized")
        print("✅ Label builder initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Label builder error: {e}")
        return False


def test_model_evaluator():
    """Test model evaluator."""
    print("\n📊 Testing model evaluator...")
    try:
        # Use the global config object directly
        evaluator = ModelEvaluator(config)
        print(f"  - Model evaluator initialized")
        print("✅ Model evaluator initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Model evaluator error: {e}")
        return False


def create_sample_data():
    """Create sample data for testing."""
    print("\n📊 Creating sample data...")
    
    # Create sample token data
    np.random.seed(42)
    n_tokens = 100
    n_timestamps = 24  # 24 hours
    
    timestamps = pd.date_range('2024-01-01', periods=n_timestamps, freq='H')
    
    sample_data = []
    for token_id in range(n_tokens):
        for timestamp in timestamps:
            sample_data.append({
                'token_id': f'token_{token_id}',
                'timestamp': timestamp,
                'price': np.random.uniform(0.1, 10.0),
                'volume_24h': np.random.uniform(1000, 100000),
                'liquidity_usd': np.random.uniform(5000, 50000),
                'transaction_count': np.random.randint(10, 1000),
                'holder_count': np.random.randint(100, 10000),
                'volatility_24h': np.random.uniform(0.1, 2.0)
            })
    
    df = pd.DataFrame(sample_data)
    print(f"  - Created {len(df)} sample data points")
    print(f"  - {n_tokens} tokens, {n_timestamps} timestamps")
    return df


def test_feature_computation(sample_data):
    """Test feature computation with sample data."""
    print("\n⚙️ Testing feature computation...")
    try:
        # Use the global config object directly
        engineer = FeatureEngineer(config)
        store = FeatureStore()  # Use default config
        
        # Compute features for a few tokens
        sample_tokens = sample_data['token_id'].unique()[:5]
        sample_data_subset = sample_data[sample_data['token_id'].isin(sample_tokens)]
        
        # Get the first timestamp for testing
        first_timestamp = sample_data_subset['timestamp'].iloc[0]
        first_token = sample_tokens[0]
        
        features = engineer.compute_features(sample_data_subset, first_token, first_timestamp)
        
        print(f"  - Computed features for {len(sample_tokens)} tokens")
        print(f"  - Feature shape: {features.shape}")
        print("✅ Feature computation successful")
        return True
    except Exception as e:
        print(f"❌ Feature computation error: {e}")
        return False


def test_label_building(sample_data):
    """Test label building with sample data."""
    print("\n🏷️ Testing label building...")
    try:
        # LabelBuilder expects a dictionary with .get() method
        builder = LabelBuilder(config.model_dump())
        
        # Build labels for a few tokens
        sample_tokens = sample_data['token_id'].unique()[:5]
        labels = builder.build_all_labels(sample_data[sample_data['token_id'].isin(sample_tokens)])
        
        print(f"  - Built labels for {len(sample_tokens)} tokens")
        print(f"  - Label shape: {labels.shape}")
        print("✅ Label building successful")
        return True
    except Exception as e:
        print(f"❌ Label building error: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Pump.fun Token Ranking System Tests")
    print("=" * 60)
    
    # Test core components
    tests = [
        test_config,
        test_data_connector,
        test_data_storage,
        test_feature_engineers,
        test_feature_store,
        test_model_registry,
        test_token_ranking_model,
        test_label_builder,
        test_model_evaluator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    # Test with sample data
    print("\n" + "=" * 60)
    print("🧪 Testing with sample data...")
    
    sample_data = create_sample_data()
    
    data_tests = [
        lambda: test_feature_computation(sample_data),
        lambda: test_label_building(sample_data)
    ]
    
    for test in data_tests:
        try:
            if test():
                passed += 1
            total += 1
        except Exception as e:
            print(f"❌ Data test failed with exception: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
