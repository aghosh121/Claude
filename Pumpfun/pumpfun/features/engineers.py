"""
Feature engineers for computing different types of features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import re

from ..config import config


class BaseFeatureEngineer(ABC):
    """Base class for feature engineers."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def compute_features(self, data: pd.DataFrame, token_id: str, timestamp: datetime) -> pd.DataFrame:
        """Compute features from raw data."""
        pass
    
    @abstractmethod
    def get_data_sources(self) -> List[str]:
        """Get list of data sources used by this engineer."""
        pass
    
    def _validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that required columns are present in data."""
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            self.logger.warning(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str = 'fill_zero') -> pd.DataFrame:
        """Handle missing values in the data."""
        if strategy == 'drop':
            return data.dropna()
        elif strategy == 'fill_zero':
            return data.fillna(0)
        elif strategy == 'fill_mean':
            return data.fillna(data.mean())
        else:
            return data


class ChainFlowEngineer(BaseFeatureEngineer):
    """Engineer for chain flow features."""
    
    def __init__(self):
        super().__init__()
        # Safely access config values with fallbacks
        try:
            if hasattr(config, 'feature') and hasattr(config.feature, 'chain_flow_windows'):
                self.windows = config.feature.chain_flow_windows
            else:
                self.windows = [1, 5, 15, 60]  # Default windows
        except Exception:
            self.windows = [1, 5, 15, 60]  # Default windows
            
        try:
            if hasattr(config, 'feature') and hasattr(config.feature, 'holder_concentration_buckets'):
                self.holder_buckets = config.feature.holder_concentration_buckets
            else:
                self.holder_buckets = [0.1, 0.25, 0.5, 0.75, 0.9]  # Default buckets
        except Exception:
            self.holder_buckets = [0.1, 0.25, 0.5, 0.75, 0.9]  # Default buckets
    
    def compute_features(self, data: pd.DataFrame, token_id: str, timestamp: datetime) -> pd.DataFrame:
        """Compute chain flow features."""
        if data.empty:
            return pd.DataFrame()
        
        features = {}
        
        try:
            # Basic transaction metrics
            if 'transaction_count' in data.columns:
                for window in self.windows:
                    features[f'txn_count_{window}m'] = data['transaction_count'].rolling(window).sum().iloc[-1]
                    features[f'txn_rate_{window}m'] = data['transaction_count'].rolling(window).mean().iloc[-1]
            
            # Volume features
            if 'buy_volume' in data.columns and 'sell_volume' in data.columns:
                for window in self.windows:
                    buy_vol = data['buy_volume'].rolling(window).sum()
                    sell_vol = data['sell_volume'].rolling(window).sum()
                    
                    features[f'buy_volume_{window}m'] = buy_vol.iloc[-1]
                    features[f'sell_volume_{window}m'] = sell_vol.iloc[-1]
                    features[f'volume_imbalance_{window}m'] = (buy_vol - sell_vol).iloc[-1]
                    features[f'buy_sell_ratio_{window}m'] = (buy_vol / (sell_vol + 1e-8)).iloc[-1]
            
            # Holder features
            if 'holder_count' in data.columns:
                for window in self.windows:
                    features[f'holder_count_{window}m'] = data['holder_count'].rolling(window).mean().iloc[-1]
                    features[f'holder_growth_{window}m'] = data['holder_count'].pct_change(window).iloc[-1]
            
            # Concentration features
            if 'top_holder_percentage' in data.columns:
                for window in self.windows:
                    features[f'top_holder_pct_{window}m'] = data['top_holder_percentage'].rolling(window).mean().iloc[-1]
                    
                    # Create concentration buckets
                    for bucket in self.holder_buckets:
                        features[f'concentration_above_{bucket}_{window}m'] = (
                            data['top_holder_percentage'] > bucket
                        ).rolling(window).sum().iloc[-1]
            
            # Velocity and acceleration features
            if 'transaction_count' in data.columns:
                for window in self.windows:
                    # First derivative (velocity)
                    features[f'txn_velocity_{window}m'] = data['transaction_count'].diff(window).iloc[-1]
                    
                    # Second derivative (acceleration)
                    features[f'txn_acceleration_{window}m'] = data['transaction_count'].diff(window).diff(window).iloc[-1]
            
            # Burstiness metrics
            if 'transaction_count' in data.columns:
                for window in self.windows:
                    window_data = data['transaction_count'].rolling(window)
                    mean_val = window_data.mean().iloc[-1]
                    std_val = window_data.std().iloc[-1]
                    
                    if mean_val > 0:
                        features[f'burstiness_{window}m'] = (std_val - mean_val) / (std_val + mean_val)
                    else:
                        features[f'burstiness_{window}m'] = 0
            
            # Unique wallet features
            if 'unique_wallets' in data.columns:
                for window in self.windows:
                    features[f'unique_wallets_{window}m'] = data['unique_wallets'].rolling(window).mean().iloc[-1]
                    features[f'wallet_churn_{window}m'] = data['unique_wallets'].pct_change(window).iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error computing chain flow features: {e}")
        
        return pd.DataFrame([features])
    
    def get_data_sources(self) -> List[str]:
        """Get data sources used by this engineer."""
        return ['solana']
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this engineer."""
        feature_names = []
        for window in self.windows:
            feature_names.extend([
                f'txn_count_{window}m',
                f'txn_rate_{window}m',
                f'buy_volume_{window}m',
                f'sell_volume_{window}m',
                f'volume_imbalance_{window}m',
                f'buy_sell_ratio_{window}m',
                f'holder_count_{window}m',
                f'holder_growth_{window}m',
                f'top_holder_pct_{window}m'
            ])
            
            for bucket in self.holder_buckets:
                feature_names.append(f'concentration_above_{bucket}_{window}m')
                
        return feature_names


class WalletQualityEngineer(BaseFeatureEngineer):
    """Engineer for wallet quality features."""
    
    def __init__(self):
        super().__init__()
        # Safely access config values with fallbacks
        try:
            if hasattr(config, 'feature') and hasattr(config.feature, 'wallet_quality_lookback_days'):
                self.lookback_days = config.feature.wallet_quality_lookback_days
            else:
                self.lookback_days = 30  # Default lookback
        except Exception:
            self.lookback_days = 30  # Default lookback
            
        try:
            if hasattr(config, 'feature') and hasattr(config.feature, 'bot_detection_threshold'):
                self.bot_threshold = config.feature.bot_detection_threshold
            else:
                self.bot_threshold = 0.7  # Default threshold
        except Exception:
            self.bot_threshold = 0.7  # Default threshold
    
    def compute_features(self, data: pd.DataFrame, token_id: str, timestamp: datetime) -> pd.DataFrame:
        """Compute wallet quality features."""
        if data.empty:
            return pd.DataFrame()
        
        features = {}
        
        try:
            # Wallet age features (if available)
            if 'wallet_age_days' in data.columns:
                features['avg_wallet_age'] = data['wallet_age_days'].mean()
                features['median_wallet_age'] = data['wallet_age_days'].median()
                features['new_wallet_ratio'] = (data['wallet_age_days'] < 1).mean()
                features['old_wallet_ratio'] = (data['wallet_age_days'] > 30).mean()
            
            # Transaction pattern features
            if 'transaction_count' in data.columns:
                features['avg_txn_per_wallet'] = data['transaction_count'].mean()
                features['txn_volatility'] = data['transaction_count'].std()
                features['high_freq_wallet_ratio'] = (data['transaction_count'] > 100).mean()
            
            # Volume pattern features
            if 'buy_volume' in data.columns and 'sell_volume' in data.columns:
                total_volume = data['buy_volume'] + data['sell_volume']
                features['avg_volume_per_wallet'] = total_volume.mean()
                features['volume_volatility'] = total_volume.std()
                features['whale_wallet_ratio'] = (total_volume > total_volume.quantile(0.9)).mean()
            
            # Bot detection features (heuristic-based)
            bot_score = self._compute_bot_score(data)
            features['bot_likelihood'] = bot_score
            features['bot_wallet_ratio'] = (bot_score > self.bot_threshold).mean()
            
            # Wallet diversity features
            if 'unique_wallets' in data.columns and 'holder_count' in data.columns:
                features['wallet_diversity'] = data['unique_wallets'] / (data['holder_count'] + 1e-8)
            
            # Historical performance features (placeholder - would need historical data)
            features['estimated_prior_pnl'] = 0.0  # Placeholder
            features['returning_alpha_wallets'] = 0.0  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error computing wallet quality features: {e}")
        
        return pd.DataFrame([features])
    
    def _compute_bot_score(self, data: pd.DataFrame) -> float:
        """Compute bot likelihood score for wallets."""
        # This is a simplified heuristic - in production, this would be a learned model
        bot_indicators = []
        
        if 'transaction_count' in data.columns:
            # High transaction frequency
            high_freq = (data['transaction_count'] > 1000).mean()
            bot_indicators.append(high_freq)
        
        if 'wallet_age_days' in data.columns:
            # Very new wallets
            new_wallets = (data['wallet_age_days'] < 0.1).mean()
            bot_indicators.append(new_wallets)
        
        if 'buy_volume' in data.columns and 'sell_volume' in data.columns:
            # Perfect timing (would need more sophisticated analysis)
            volume_timing = 0.5  # Placeholder
            bot_indicators.append(volume_timing)
        
        # Average of indicators
        return np.mean(bot_indicators) if bot_indicators else 0.0
    
    def get_data_sources(self) -> List[str]:
        """Get data sources used by this engineer."""
        return ['solana']
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this engineer."""
        return [
            'wallet_age_days',
            'transaction_count_24h',
            'unique_tokens_held',
            'total_value_locked',
            'buy_sell_ratio_24h',
            'volume_consistency',
            'bot_score',
            'wallet_reputation_score',
            'risk_level'
        ]


class SocialEngineer(BaseFeatureEngineer):
    """Engineer for social media features."""
    
    def __init__(self):
        super().__init__()
        # Safely access config values with fallbacks
        try:
            if hasattr(config, 'feature') and hasattr(config.feature, 'social_platforms'):
                self.platforms = config.feature.social_platforms
            else:
                self.platforms = ["twitter", "telegram", "discord"]  # Default platforms
        except Exception:
            self.platforms = ["twitter", "telegram", "discord"]  # Default platforms
            
        try:
            if hasattr(config, 'feature') and hasattr(config.feature, 'sentiment_lookback_hours'):
                self.sentiment_lookback = config.feature.sentiment_lookback_hours
            else:
                self.sentiment_lookback = 24  # Default lookback
        except Exception:
            self.sentiment_lookback = 24  # Default lookback
    
    def compute_features(self, data: pd.DataFrame, token_id: str, timestamp: datetime) -> pd.DataFrame:
        """Compute social media features."""
        if data.empty:
            return pd.DataFrame()
        
        features = {}
        
        try:
            # Platform-specific features
            for platform in self.platforms:
                platform_data = data[data['platform'] == platform]
                
                if not platform_data.empty:
                    # Mention volume
                    features[f'{platform}_mention_count'] = platform_data['mention_count'].sum()
                    features[f'{platform}_mention_rate'] = platform_data['mention_count'].mean()
                    
                    # Engagement metrics
                    if 'engagement_rate' in platform_data.columns:
                        features[f'{platform}_avg_engagement'] = platform_data['engagement_rate'].mean()
                        features[f'{platform}_engagement_volatility'] = platform_data['engagement_rate'].std()
                    
                    # Account quality
                    if 'account_age_days' in platform_data.columns:
                        features[f'{platform}_avg_account_age'] = platform_data['account_age_days'].mean()
                        features[f'{platform}_new_account_ratio'] = (platform_data['account_age_days'] < 1).mean()
                    
                    if 'follower_count' in platform_data.columns:
                        features[f'{platform}_avg_followers'] = platform_data['follower_count'].mean()
                        features[f'{platform}_verified_account_ratio'] = (platform_data['follower_count'] > 10000).mean()
            
            # Cross-platform features
            total_mentions = data['mention_count'].sum()
            features['total_social_mentions'] = total_mentions
            features['social_platforms_active'] = data['platform'].nunique()
            
            # Sentiment features
            if 'sentiment_score' in data.columns:
                features['avg_sentiment'] = data['sentiment_score'].mean()
                features['sentiment_volatility'] = data['sentiment_score'].std()
                features['positive_sentiment_ratio'] = (data['sentiment_score'] > 0.5).mean()
                features['negative_sentiment_ratio'] = (data['sentiment_score'] < -0.5).mean()
            
            # Velocity features
            if 'timestamp' in data.columns:
                # Mentions per hour
                data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
                hourly_mentions = data.groupby('hour')['mention_count'].sum()
                features['peak_hour_mentions'] = hourly_mentions.max()
                features['mention_hour_volatility'] = hourly_mentions.std()
            
            # Credibility-weighted engagement
            if 'engagement_rate' in data.columns and 'follower_count' in data.columns:
                # Weight engagement by follower count (log scale to handle large numbers)
                credibility_weight = np.log1p(data['follower_count'])
                weighted_engagement = (data['engagement_rate'] * credibility_weight).sum()
                features['credibility_weighted_engagement'] = weighted_engagement
            
        except Exception as e:
            self.logger.error(f"Error computing social features: {e}")
        
        return pd.DataFrame([features])
    
    def get_data_sources(self) -> List[str]:
        """Get data sources used by this engineer."""
        return ['social']
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this engineer."""
        feature_names = []
        
        for platform in self.platforms:
            feature_names.extend([
                f'{platform}_mention_count',
                f'{platform}_mention_rate',
                f'{platform}_avg_engagement',
                f'{platform}_engagement_volatility',
                f'{platform}_avg_account_age',
                f'{platform}_new_account_ratio',
                f'{platform}_avg_followers',
                f'{platform}_verified_account_ratio'
            ])
        
        feature_names.extend([
            'total_social_mentions',
            'social_platforms_active',
            'avg_sentiment',
            'sentiment_volatility',
            'positive_sentiment_ratio',
            'negative_sentiment_ratio',
            'peak_hour_mentions',
            'mention_hour_volatility',
            'credibility_weighted_engagement'
        ])
        
        return feature_names


class ContentEngineer(BaseFeatureEngineer):
    """Engineer for content and NLP features."""
    
    def __init__(self):
        super().__init__()
        # Safely access config values with fallbacks
        try:
            if hasattr(config, 'feature') and hasattr(config.feature, 'content_max_length'):
                self.max_length = config.feature.content_max_length
            else:
                self.max_length = 1000  # Default max length
        except Exception:
            self.max_length = 1000  # Default max length
            
        try:
            if hasattr(config, 'feature') and hasattr(config.feature, 'nlp_embedding_model'):
                self.embedding_model = config.feature.nlp_embedding_model
            else:
                self.embedding_model = "all-MiniLM-L6-v2"  # Default model
        except Exception:
            self.embedding_model = "all-MiniLM-L6-v2"  # Default model
    
    def compute_features(self, data: pd.DataFrame, token_id: str, timestamp: datetime) -> pd.DataFrame:
        """Compute content and NLP features."""
        if data.empty:
            return pd.DataFrame()
        
        features = {}
        
        try:
            # Text length features
            if 'description' in data.columns:
                descriptions = data['description'].fillna('')
                features['avg_description_length'] = descriptions.str.len().mean()
                features['max_description_length'] = descriptions.str.len().max()
                features['description_length_std'] = descriptions.str.len().std()
                
                # Truncate for processing
                descriptions = descriptions.str[:self.max_length]
            
            # Readability features
            if 'description' in data.columns:
                features['avg_readability_score'] = self._compute_readability(descriptions)
                features['complexity_score'] = self._compute_complexity(descriptions)
            
            # Keyphrase features
            if 'description' in data.columns:
                keyphrase_features = self._extract_keyphrases(descriptions)
                features.update(keyphrase_features)
            
            # Sentiment and emotion features
            if 'description' in data.columns:
                sentiment_features = self._analyze_sentiment(descriptions)
                features.update(sentiment_features)
            
            # Specificity features
            if 'description' in data.columns:
                features['content_specificity'] = self._compute_specificity(descriptions)
                features['content_entropy'] = self._compute_entropy(descriptions)
            
            # Link and verification features
            if 'verified_links' in data.columns:
                features['verified_links_present'] = data['verified_links'].any()
                features['verified_links_count'] = data['verified_links'].sum()
            
            # Content structure features
            if 'description' in data.columns:
                features['has_emoji'] = descriptions.str.contains(r'[^\w\s]').mean()
                features['has_hashtags'] = descriptions.str.contains(r'#').mean()
                features['has_mentions'] = descriptions.str.contains(r'@').mean()
                features['has_urls'] = descriptions.str.contains(r'http').mean()
            
        except Exception as e:
            self.logger.error(f"Error computing content features: {e}")
        
        return pd.DataFrame([features])
    
    def _compute_readability(self, texts: pd.Series) -> float:
        """Compute average readability score."""
        # Simplified Flesch-Kincaid grade level
        try:
            scores = []
            for text in texts:
                if pd.isna(text) or len(text) == 0:
                    continue
                
                sentences = len(re.split(r'[.!?]+', text))
                words = len(text.split())
                syllables = len(re.findall(r'[aeiouy]+', text.lower()))
                
                if sentences > 0 and words > 0:
                    score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
                    scores.append(max(0, min(100, score)))
            
            return np.mean(scores) if scores else 0.0
        except:
            return 0.0
    
    def _compute_complexity(self, texts: pd.Series) -> float:
        """Compute text complexity score."""
        try:
            scores = []
            for text in texts:
                if pd.isna(text) or len(text) == 0:
                    continue
                
                words = text.split()
                if len(words) == 0:
                    continue
                
                # Average word length
                avg_word_length = np.mean([len(word) for word in words])
                scores.append(avg_word_length)
            
            return np.mean(scores) if scores else 0.0
        except:
            return 0.0
    
    def _extract_keyphrases(self, texts: pd.Series) -> Dict[str, float]:
        """Extract keyphrase features."""
        features = {}
        
        try:
            # Common crypto/meme terms
            crypto_terms = ['moon', 'pump', 'diamond', 'hands', 'rocket', 'lambo', 'mooning']
            meme_terms = ['ape', 'hodl', 'fud', 'fomo', 'shill', 'moonboy']
            
            for term in crypto_terms:
                features[f'contains_{term}'] = texts.str.contains(term, case=False).mean()
            
            for term in meme_terms:
                features[f'contains_{term}'] = texts.str.contains(term, case=False).mean()
            
            # Overall crypto/meme density
            all_terms = crypto_terms + meme_terms
            term_counts = texts.str.count('|'.join(all_terms), case=False)
            features['crypto_meme_density'] = term_counts.mean()
            
        except Exception as e:
            self.logger.warning(f"Error extracting keyphrases: {e}")
        
        return features
    
    def _analyze_sentiment(self, texts: pd.Series) -> Dict[str, float]:
        """Analyze sentiment and emotion."""
        features = {}
        
        try:
            # Simple sentiment analysis
            positive_words = ['good', 'great', 'amazing', 'awesome', 'bullish', 'moon']
            negative_words = ['bad', 'terrible', 'awful', 'bearish', 'dump']
            
            positive_counts = texts.str.count('|'.join(positive_words), case=False)
            negative_counts = texts.str.count('|'.join(negative_words), case=False)
            
            features['positive_sentiment'] = positive_counts.mean()
            features['negative_sentiment'] = negative_counts.mean()
            features['sentiment_balance'] = (positive_counts - negative_counts).mean()
            
        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment: {e}")
        
        return features
    
    def _compute_specificity(self, texts: pd.Series) -> float:
        """Compute content specificity score."""
        try:
            # Count unique words vs total words
            total_words = sum(len(text.split()) for text in texts if pd.notna(text))
            unique_words = len(set(word.lower() for text in texts if pd.notna(text) for word in text.split()))
            
            if total_words > 0:
                return unique_words / total_words
            return 0.0
        except:
            return 0.0
    
    def _compute_entropy(self, texts: pd.Series) -> float:
        """Compute content entropy."""
        try:
            # Simple character-level entropy
            all_text = ' '.join(texts.fillna(''))
            if len(all_text) == 0:
                return 0.0
            
            char_counts = pd.Series(list(all_text)).value_counts()
            probs = char_counts / len(all_text)
            entropy = -np.sum(probs * np.log2(probs))
            
            return entropy
        except:
            return 0.0
    
    def get_data_sources(self) -> List[str]:
        """Get data sources used by this engineer."""
        return ['content', 'nlp']
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this engineer."""
        return [
            'avg_text_length',
            'text_length_std',
            'readability_score',
            'complexity_score',
            'specificity_score',
            'entropy_score',
            'keyphrase_count',
            'avg_sentiment_score',
            'sentiment_volatility',
            'positive_content_ratio',
            'negative_content_ratio',
            'neutral_content_ratio',
            'content_quality_score',
            'engagement_potential_score'
        ]


class ImageEngineer(BaseFeatureEngineer):
    """Engineer for image and media features."""
    
    def __init__(self):
        super().__init__()
        # Safely access config values with fallbacks
        try:
            if hasattr(config, 'feature') and hasattr(config.feature, 'image_max_size'):
                self.max_size = config.feature.image_max_size
            else:
                self.max_size = 512  # Default max size
        except Exception:
            self.max_size = 512  # Default max size
            
        try:
            if hasattr(config, 'feature') and hasattr(config.feature, 'image_embedding_model'):
                self.embedding_model = config.feature.image_embedding_model
            else:
                self.embedding_model = "clip-vit-base-patch32"  # Default model
        except Exception:
            self.embedding_model = "clip-vit-base-patch32"  # Default model
    
    def compute_features(self, data: pd.DataFrame, token_id: str, timestamp: datetime) -> pd.DataFrame:
        """Compute image and media features."""
        if data.empty:
            return pd.DataFrame()
        
        features = {}
        
        try:
            # Basic image properties (if available)
            if 'image_url' in data.columns:
                features['has_image'] = data['image_url'].notna().any()
                features['image_count'] = data['image_url'].notna().sum()
            
            # Image format features
            if 'image_format' in data.columns:
                format_counts = data['image_format'].value_counts()
                features['png_ratio'] = format_counts.get('png', 0) / len(data)
                features['jpg_ratio'] = format_counts.get('jpg', 0) / len(data)
                features['gif_ratio'] = format_counts.get('gif', 0) / len(data)
            
            # Image size features
            if 'image_width' in data.columns and 'image_height' in data.columns:
                features['avg_image_width'] = data['image_width'].mean()
                features['avg_image_height'] = data['image_height'].mean()
                features['avg_aspect_ratio'] = (data['image_width'] / data['image_height']).mean()
                features['image_size_std'] = (data['image_width'] * data['image_height']).std()
            
            # Compression and quality features
            if 'file_size' in data.columns:
                features['avg_file_size'] = data['file_size'].mean()
                features['file_size_std'] = data['file_size'].std()
                
                # Compression ratio (if dimensions available)
                if 'image_width' in data.columns and 'image_height' in data.columns:
                    pixel_counts = data['image_width'] * data['image_height']
                    compression_ratio = data['file_size'] / (pixel_counts * 3)  # Assume 3 bytes per pixel
                    features['avg_compression_ratio'] = compression_ratio.mean()
            
            # Content detection features (placeholder - would use actual ML models)
            features['face_detected'] = 0.0  # Placeholder
            features['object_detected'] = 0.0  # Placeholder
            features['text_in_image'] = 0.0  # Placeholder
            features['nsfw_likelihood'] = 0.0  # Placeholder
            
            # Meme classification features (placeholder)
            features['meme_class_pepe'] = 0.0
            features['meme_class_doge'] = 0.0
            features['meme_class_wojak'] = 0.0
            features['meme_class_other'] = 0.0
            
            # Aesthetic and originality features
            features['aesthetic_score'] = 0.5  # Placeholder
            features['originality_score'] = 0.5  # Placeholder
            features['near_duplicate_count'] = 0.0  # Placeholder
            
            # Animation features
            if 'is_animated' in data.columns:
                features['animation_ratio'] = data['is_animated'].mean()
            else:
                features['animation_ratio'] = 0.0
            
        except Exception as e:
            self.logger.error(f"Error computing image features: {e}")
        
        return pd.DataFrame([features])
    
    def get_data_sources(self) -> List[str]:
        """Get data sources used by this engineer."""
        return ['image', 'vision']
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this engineer."""
        return [
            'image_quality_score',
            'brightness_score',
            'contrast_score',
            'sharpness_score',
            'color_variety_score',
            'text_detection_score',
            'face_detection_score',
            'object_detection_score',
            'brand_safety_score',
            'aesthetic_appeal_score',
            'memorability_score',
            'engagement_potential_score',
            'image_complexity_score',
            'visual_consistency_score'
        ]


class RegimeEngineer(BaseFeatureEngineer):
    """Engineer for market regime features."""
    
    def __init__(self):
        super().__init__()
        self.volatility_buckets = config.feature.regime_volatility_buckets
    
    def compute_features(self, data: Dict[str, pd.DataFrame], token_id: str, timestamp: datetime) -> pd.DataFrame:
        """Compute market regime features."""
        features = {}
        
        try:
            # Time-based features
            features['hour_of_day'] = timestamp.hour
            features['day_of_week'] = timestamp.weekday()
            features['is_weekend'] = timestamp.weekday() >= 5
            features['is_business_hours'] = 9 <= timestamp.hour <= 17
            
            # Market hours features (crypto markets are 24/7, but activity varies)
            features['is_peak_hours'] = timestamp.hour in [0, 8, 16]  # UTC peak times
            features['is_low_activity_hours'] = timestamp.hour in [4, 12, 20]  # Low activity times
            
            # Seasonal features
            features['month'] = timestamp.month
            features['quarter'] = (timestamp.month - 1) // 3 + 1
            features['is_month_end'] = timestamp.day >= 25
            features['is_quarter_end'] = timestamp.day >= 25 and timestamp.month in [3, 6, 9, 12]
            
            # Volatility regime features (if volatility data available)
            if 'solana' in data and not data['solana'].empty:
                solana_data = data['solana']
                
                # Compute rolling volatility if price data available
                if 'price' in solana_data.columns:
                    price_returns = solana_data['price'].pct_change()
                    volatility_1h = price_returns.rolling(60).std().iloc[-1] if len(price_returns) >= 60 else 0
                    volatility_24h = price_returns.rolling(1440).std().iloc[-1] if len(price_returns) >= 1440 else 0
                    
                    features['solana_volatility_1h'] = volatility_1h
                    features['solana_volatility_24h'] = volatility_24h
                    
                    # Volatility regime classification
                    for i, bucket in enumerate(self.volatility_buckets):
                        features[f'volatility_regime_{i}'] = 1.0 if volatility_24h <= bucket else 0.0
            
            # Network fee features (if available)
            if 'solana' in data and not data['solana'].empty:
                solana_data = data['solana']
                if 'network_fees' in solana_data.columns:
                    features['avg_network_fees'] = solana_data['network_fees'].mean()
                    features['network_fees_std'] = solana_data['network_fees'].std()
                    features['high_fee_period'] = (solana_data['network_fees'] > solana_data['network_fees'].quantile(0.9)).mean()
            
            # Memecoin index features (placeholder - would need actual index data)
            features['memecoin_index_level'] = 0.5  # Placeholder
            features['memecoin_index_volatility'] = 0.1  # Placeholder
            
            # Global market sentiment (placeholder)
            features['global_sentiment_bullish'] = 0.5
            features['global_sentiment_bearish'] = 0.5
            features['global_fear_greed_index'] = 50.0
            
            # Interaction terms
            features['hour_volatility_interaction'] = features.get('hour_of_day', 0) * features.get('solana_volatility_1h', 0)
            features['weekend_volatility_interaction'] = features.get('is_weekend', 0) * features.get('solana_volatility_24h', 0)
            
        except Exception as e:
            self.logger.error(f"Error computing regime features: {e}")
        
        return pd.DataFrame([features])
    
    def get_data_sources(self) -> List[str]:
        """Get data sources used by this engineer."""
        return ['solana', 'pumpfun']  # Regime features can use multiple sources
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this engineer."""
        return [
            'volatility_regime',
            'volume_regime',
            'market_regime',
            'trend_strength',
            'regime_confidence',
            'regime_transition_probability',
            'regime_persistence_score',
            'regime_volatility_score',
            'regime_volume_score',
            'regime_trend_score'
        ]


class FeatureEngineer:
    """Main feature engineering orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engineers = {
            'chain_flow': ChainFlowEngineer(),
            'wallet_quality': WalletQualityEngineer(),
            'social': SocialEngineer(),
            'content': ContentEngineer(),
            'image': ImageEngineer(),
            'regime': RegimeEngineer()
        }
        
    def compute_features(self, data: pd.DataFrame, token_id: str, timestamp: datetime) -> pd.DataFrame:
        """Compute all features for a token."""
        all_features = {}
        
        for family_name, engineer in self.engineers.items():
            if family_name in self.config.feature.feature_families:
                try:
                    features = engineer.compute_features(data, token_id, timestamp)
                    if not features.empty:
                        all_features[f"{family_name}_"] = features
                except Exception as e:
                    logging.warning(f"Failed to compute {family_name} features: {e}")
                    continue
        
        if not all_features:
            return pd.DataFrame()
            
        # Combine all features
        combined_features = pd.concat(all_features.values(), axis=1)
        combined_features['token_id'] = token_id
        combined_features['timestamp'] = timestamp
        
        return combined_features
        
    def get_feature_names(self) -> List[str]:
        """Get all available feature names."""
        feature_names = []
        for engineer in self.engineers.values():
            feature_names.extend(engineer.get_feature_names())
        return feature_names
        
    def get_data_sources(self) -> List[str]:
        """Get all data sources used by feature engineers."""
        data_sources = set()
        for engineer in self.engineers.values():
            data_sources.update(engineer.get_data_sources())
        return list(data_sources)
