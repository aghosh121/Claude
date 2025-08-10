"""
Command-line interface for the Pump.fun token ranking system.

This module provides a comprehensive CLI for interacting with all system
components including data management, model training, evaluation, and serving.
"""

import click
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import asyncio
import sys

from .config import config
from .data.connector import DataConnector
from .data.storage import DataStorage
from .data.snapshotter import DataSnapshotter
from .features.engineers import FeatureEngineer
from .features.store import FeatureStore
from .models import ModelRegistry
from .evaluation import ModelEvaluator
from .labels import LabelBuilder
from .serve import ModelServer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Path to config file')
def cli(verbose, config_file):
    """Pump.fun Token Ranking System CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if config_file:
        # Load custom config
        click.echo(f"Loading config from {config_file}")
    
    click.echo("🚀 Pump.fun Token Ranking System")


@cli.group()
def data():
    """Data management commands"""
    pass


@data.command()
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--limit', '-l', type=int, default=1000, help='Number of records to fetch')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', 'output_format', type=click.Choice(['csv', 'json', 'parquet']), default='csv')
def fetch(source, limit, output, output_format):
    """Fetch data from a source"""
    try:
        connector = DataConnector(config.dict())
        data = connector.fetch_data(source, limit=limit)
        
        if data.empty:
            click.echo(f"No data found for source: {source}")
            return
        
        click.echo(f"Fetched {len(data)} records from {source}")
        
        if output:
            if output_format == 'csv':
                data.to_csv(output, index=False)
            elif output_format == 'json':
                data.to_json(output, orient='records', indent=2)
            elif output_format == 'parquet':
                data.to_parquet(output, index=False)
            click.echo(f"Data saved to {output}")
        else:
            # Display sample
            click.echo("\nSample data:")
            click.echo(data.head().to_string())
            
    except Exception as e:
        click.echo(f"Error fetching data: {e}", err=True)
        sys.exit(1)


@data.command()
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--storage-type', type=click.Choice(['raw', 'processed']), default='raw')
@click.option('--compression', type=click.Choice(['snappy', 'gzip', 'brotli']), default='snappy')
def store(source, storage_type, compression):
    """Store data from a source"""
    try:
        # Fetch and store data
        connector = DataConnector(config.dict())
        storage = DataStorage(config.dict())
        
        data = connector.fetch_data(source)
        if data.empty:
            click.echo(f"No data found for source: {source}")
            return
        
        metadata = storage.store_data(source, data, storage_type=storage_type, compression=compression)
        
        click.echo(f"Stored {len(data)} records from {source}")
        click.echo(f"File size: {metadata.file_size_bytes / 1024:.2f} KB")
        click.echo(f"Compression ratio: {metadata.compression_ratio:.2%}")
        
    except Exception as e:
        click.echo(f"Error storing data: {e}", err=True)
        sys.exit(1)


@data.command()
@click.option('--source', '-s', help='Data source name (optional)')
@click.option('--storage-type', type=click.Choice(['raw', 'processed']), default='raw')
def list_sources(source, storage_type):
    """List available data sources"""
    try:
        storage = DataStorage(config.dict())
        
        if source:
            info = storage.get_data_info(source, storage_type)
            if info:
                click.echo(f"\nData source: {source}")
                click.echo(f"Total files: {info['total_files']}")
                click.echo(f"Total size: {info['total_size_gb']:.2f} GB")
                click.echo(f"Total records: {info['total_records']}")
                click.echo(f"Latest: {info['latest_timestamp']}")
                click.echo(f"Oldest: {info['oldest_timestamp']}")
                
                click.echo("\nFile details:")
                for file_info in info['file_info'][:5]:  # Show first 5 files
                    click.echo(f"  {file_info['filename']} - {file_info['size_bytes']} bytes - {file_info['age_hours']:.1f} hours old")
            else:
                click.echo(f"No data found for source: {source}")
        else:
            sources = storage.list_sources(storage_type)
            if sources:
                click.echo(f"Available {storage_type} data sources:")
                for source_name in sources:
                    info = storage.get_data_info(source_name, storage_type)
                    if info:
                        click.echo(f"  {source_name}: {info['total_files']} files, {info['total_size_gb']:.2f} GB")
            else:
                click.echo(f"No {storage_type} data sources found")
                
    except Exception as e:
        click.echo(f"Error listing sources: {e}", err=True)
        sys.exit(1)


@data.command()
@click.option('--max-age-hours', type=int, default=168, help='Maximum age in hours to keep')
@click.option('--storage-type', type=click.Choice(['raw', 'processed']), default='raw')
@click.option('--dry-run', is_flag=True, help='Show what would be deleted without actually deleting')
def cleanup(max_age_hours, storage_type, dry_run):
    """Clean up old data files"""
    try:
        storage = DataStorage(config.dict())
        
        if dry_run:
            click.echo(f"DRY RUN: Would clean up data older than {max_age_hours} hours")
            # This would need to be implemented in the storage class
            click.echo("Dry run cleanup not yet implemented")
        else:
            result = storage.cleanup_old_data(max_age_hours, storage_type)
            click.echo(f"Cleanup completed: {result['files_removed']} files removed")
            click.echo(f"Freed space: {result['bytes_freed'] / (1024**3):.2f} GB")
            
    except Exception as e:
        click.echo(f"Error during cleanup: {e}", err=True)
        sys.exit(1)


@cli.group()
def features():
    """Feature engineering commands"""
    pass


@features.command()
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--storage-type', type=click.Choice(['raw', 'processed']), default='raw')
def engineer(source, output, storage_type):
    """Generate features from data"""
    try:
        # Load data
        storage = DataStorage(config.dict())
        data = storage.load_data(source, storage_type=storage_type)
        
        if data.empty:
            click.echo(f"No data found for source: {source}")
            return
        
        # Engineer features
        feature_engineer = FeatureEngineer(config.dict())
        features = feature_engineer.engineer_features(data)
        
        click.echo(f"Generated {len(features)} feature records from {len(data)} data records")
        click.echo(f"Feature columns: {', '.join(features.columns)}")
        
        if output:
            features.to_parquet(output, index=False)
            click.echo(f"Features saved to {output}")
        else:
            # Store in processed storage
            metadata = storage.store_data(f"{source}_features", features, storage_type="processed")
            click.echo(f"Features stored in processed storage")
            
    except Exception as e:
        click.echo(f"Error engineering features: {e}", err=True)
        sys.exit(1)


@features.command()
@click.option('--source', '-s', help='Data source name (optional)')
def list_features(source):
    """List available features"""
    try:
        feature_store = FeatureStore(config.dict())
        
        if source:
            features = feature_store.get_features(source)
            if not features.empty:
                click.echo(f"Features for {source}:")
                click.echo(features.head().to_string())
            else:
                click.echo(f"No features found for {source}")
        else:
            sources = feature_store.list_feature_sources()
            if sources:
                click.echo("Available feature sources:")
                for source_name in sources:
                    click.echo(f"  {source_name}")
            else:
                click.echo("No feature sources found")
                
    except Exception as e:
        click.echo(f"Error listing features: {e}", err=True)
        sys.exit(1)


@cli.group()
def models():
    """Model management commands"""
    pass


@models.command()
@click.option('--model', '-m', required=True, help='Model name')
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--test-size', type=float, default=0.2, help='Test set size')
@click.option('--random-state', type=int, default=42, help='Random state for reproducibility')
def train(model, source, target, test_size, random_state):
    """Train a model"""
    try:
        # Load data
        storage = DataStorage(config.dict())
        data = storage.load_data(source, storage_type="processed")
        
        if data.empty:
            click.echo(f"No processed data found for source: {source}")
            return
        
        if target not in data.columns:
            click.echo(f"Target column '{target}' not found in data")
            return
        
        # Prepare features and target
        X = data.drop(columns=[target])
        y = data[target]
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        click.echo(f"Training {model} with {len(X)} samples and {len(X.columns)} features")
        
        # Train model
        model_registry = ModelRegistry(config.dict())
        trained_model = model_registry.train_model(model, X, y, test_size=test_size, random_state=random_state)
        
        click.echo(f"Model {model} trained successfully")
        
        # Save model
        model_path = Path(config.model.model_save_dir) / f"{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_registry.save_model(model, model_path)
        click.echo(f"Model saved to {model_path}")
        
    except Exception as e:
        click.echo(f"Error training model: {e}", err=True)
        sys.exit(1)


@models.command()
@click.option('--model', '-m', required=True, help='Model name')
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--test-size', type=float, default=0.2, help='Test set size')
@click.option('--random-state', type=int, default=42, help='Random state for reproducibility')
def evaluate(model, source, target, test_size, random_state):
    """Evaluate a model"""
    try:
        # Load data
        storage = DataStorage(config.dict())
        data = storage.load_data(source, storage_type="processed")
        
        if data.empty:
            click.echo(f"No processed data found for source: {source}")
            return
        
        if target not in data.columns:
            click.echo(f"Target column '{target}' not found in data")
            return
        
        # Prepare features and target
        X = data.drop(columns=[target])
        y = data[target]
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        click.echo(f"Evaluating {model} with {len(X)} samples and {len(X.columns)} features")
        
        # Evaluate model
        evaluator = ModelEvaluator(config.dict())
        performance = evaluator.evaluate_model(model, X, y, test_size, random_state)
        
        click.echo(f"\nEvaluation Results for {model}:")
        click.echo(f"RMSE: {performance.metrics.rmse:.4f}")
        click.echo(f"MAE: {performance.metrics.mae:.4f}")
        click.echo(f"R² Score: {performance.metrics.r2:.4f}")
        click.echo(f"MAPE: {performance.metrics.mape:.2f}%")
        click.echo(f"Training Time: {performance.training_time:.2f} seconds")
        click.echo(f"Prediction Time: {performance.prediction_time:.4f} seconds")
        click.echo(f"Model Size: {performance.model_size_mb:.2f} MB")
        
    except Exception as e:
        click.echo(f"Error evaluating model: {e}", err=True)
        sys.exit(1)


@models.command()
@click.option('--models', '-m', required=True, help='Comma-separated list of model names')
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--test-size', type=float, default=0.2, help='Test set size')
@click.option('--random-state', type=int, default=42, help='Random state for reproducibility')
def compare(models, source, target, test_size, random_state):
    """Compare multiple models"""
    try:
        model_names = [m.strip() for m in models.split(',')]
        
        # Load data
        storage = DataStorage(config.dict())
        data = storage.load_data(source, storage_type="processed")
        
        if data.empty:
            click.echo(f"No processed data found for source: {source}")
            return
        
        if target not in data.columns:
            click.echo(f"Target column '{target}' not found in data")
            return
        
        # Prepare features and target
        X = data.drop(columns=[target])
        y = data[target]
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        click.echo(f"Comparing {len(model_names)} models with {len(X)} samples and {len(X.columns)} features")
        
        # Compare models
        evaluator = ModelEvaluator(config.dict())
        comparison = evaluator.compare_models(model_names, X, y, test_size, random_state)
        
        click.echo("\nModel Comparison Results:")
        click.echo(comparison.to_string(index=False))
        
        # Find best model
        best_model = comparison.iloc[0]
        click.echo(f"\nBest performing model: {best_model['model_name']} (RMSE: {best_model['rmse']:.4f})")
        
    except Exception as e:
        click.echo(f"Error comparing models: {e}", err=True)
        sys.exit(1)


@models.command()
def list_models():
    """List available models"""
    try:
        model_registry = ModelRegistry(config.dict())
        available_models = model_registry.list_models()
        
        if available_models:
            click.echo("Available models:")
            for model_name in available_models:
                click.echo(f"  {model_name}")
        else:
            click.echo("No models available")
            
    except Exception as e:
        click.echo(f"Error listing models: {e}", err=True)
        sys.exit(1)


@cli.group()
def labels():
    """Label building commands"""
    pass


@labels.command()
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--method', '-m', type=click.Choice(['future_return', 'ranking', 'classification']), default='future_return')
@click.option('--horizon', type=int, default=24, help='Prediction horizon in hours')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def build_labels(source, method, horizon, output):
    """Build labels for training"""
    try:
        # Load data
        storage = DataStorage(config.dict())
        data = storage.load_data(source, storage_type="raw")
        
        if data.empty:
            click.echo(f"No data found for source: {source}")
            return
        
        click.echo(f"Building {method} labels for {len(data)} records with {horizon}h horizon")
        
        # Build labels
        label_builder = LabelBuilder(config.dict())
        labels = label_builder.build_labels(data, method=method, horizon=horizon)
        
        click.echo(f"Generated {len(labels)} label records")
        
        if output:
            labels.to_parquet(output, index=False)
            click.echo(f"Labels saved to {output}")
        else:
            # Store in processed storage
            metadata = storage.store_data(f"{source}_labels", labels, storage_type="processed")
            click.echo(f"Labels stored in processed storage")
            
    except Exception as e:
        click.echo(f"Error building labels: {e}", err=True)
        sys.exit(1)


@cli.group()
def serve():
    """Model serving commands"""
    pass


@serve.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--workers', default=1, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def start(host, port, workers, reload):
    """Start the model serving API"""
    try:
        click.echo(f"Starting Pump.fun API server on {host}:{port}")
        
        server = ModelServer(config.dict())
        
        if reload:
            click.echo("Auto-reload enabled")
            # This would need to be implemented in the server class
            click.echo("Auto-reload not yet implemented")
        
        # Start server
        server.start(host=host, port=port, workers=workers)
        
    except Exception as e:
        click.echo(f"Error starting server: {e}", err=True)
        sys.exit(1)


@serve.command()
@click.option('--model', '-m', required=True, help='Model name')
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input data file')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def predict(model, input, output):
    """Make predictions using a trained model"""
    try:
        # Load input data
        if input.endswith('.csv'):
            data = pd.read_csv(input)
        elif input.endswith('.parquet'):
            data = pd.read_parquet(input)
        else:
            click.echo("Input file must be CSV or Parquet")
            return
        
        click.echo(f"Making predictions with {model} on {len(data)} records")
        
        # Load model and make predictions
        model_registry = ModelRegistry(config.dict())
        trained_model = model_registry.load_model(model)
        
        if trained_model is None:
            click.echo(f"Model {model} not found")
            return
        
        # Make predictions
        predictions = trained_model.predict(data)
        
        # Add predictions to data
        result_data = data.copy()
        result_data['prediction'] = predictions
        
        click.echo(f"Predictions completed")
        
        if output:
            if output.endswith('.csv'):
                result_data.to_csv(output, index=False)
            elif output.endswith('.parquet'):
                result_data.to_parquet(output, index=False)
            else:
                result_data.to_csv(output, index=False)
            click.echo(f"Results saved to {output}")
        else:
            # Display sample
            click.echo("\nSample predictions:")
            click.echo(result_data[['prediction']].head().to_string())
            
    except Exception as e:
        click.echo(f"Error making predictions: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show system status"""
    try:
        click.echo("🔍 Pump.fun System Status")
        click.echo("=" * 40)
        
        # Data storage status
        storage = DataStorage(config.dict())
        stats = storage.get_storage_stats()
        
        click.echo(f"📊 Data Storage:")
        click.echo(f"  Total files: {stats['total_files']}")
        click.echo(f"  Total size: {stats['total_size_gb']:.2f} GB")
        click.echo(f"  Oldest data: {stats['oldest_data']}")
        click.echo(f"  Newest data: {stats['newest_data']}")
        
        # Model status
        model_registry = ModelRegistry(config.dict())
        available_models = model_registry.list_models()
        
        click.echo(f"\n🤖 Models:")
        click.echo(f"  Available: {len(available_models)}")
        for model_name in available_models:
            click.echo(f"    - {model_name}")
        
        # Feature status
        feature_store = FeatureStore(config.dict())
        feature_sources = feature_store.list_feature_sources()
        
        click.echo(f"\n⚡ Features:")
        click.echo(f"  Sources: {len(feature_sources)}")
        for source in feature_sources:
            click.echo(f"    - {source}")
        
        click.echo("\n✅ System is running")
        
    except Exception as e:
        click.echo(f"Error getting status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def export_config(output):
    """Export current configuration"""
    try:
        config_dict = config.dict()
        
        if output:
            with open(output, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            click.echo(f"Configuration exported to {output}")
        else:
            click.echo(json.dumps(config_dict, indent=2, default=str))
            
    except Exception as e:
        click.echo(f"Error exporting config: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
