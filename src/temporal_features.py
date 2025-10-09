"""
Temporal (time-series) feature extraction for audio files.
Handles frame-based processing with adjustable window sizes and temporal context.
"""

import librosa
import numpy as np
from typing import Dict, Any, Tuple, Optional


class TemporalFeatureExtractor:
    """
    Extract time-series features from audio with temporal context.
    
    This extractor processes audio in frames (windows) and can:
    - Use configurable frame sizes (e.g., 10-25ms)
    - Compute delta features (1st derivative - velocity)
    - Compute delta-delta features (2nd derivative - acceleration)
    - Stack consecutive frames for temporal context
    """
    
    def __init__(
        self, 
        sr: int = 22050,
        n_mfcc: int = 13,
        frame_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        use_deltas: bool = True,
        use_delta_deltas: bool = True,
        context_frames: int = 0
    ):
        """
        Initialize the temporal feature extractor.
        
        Args:
            sr: Sample rate for audio loading
            n_mfcc: Number of MFCC coefficients to extract
            frame_length_ms: Frame/window length in milliseconds (default: 25ms)
            hop_length_ms: Hop length (stride) in milliseconds (default: 10ms)
            use_deltas: Include delta (velocity) features
            use_delta_deltas: Include delta-delta (acceleration) features
            context_frames: Number of frames before/after to stack (0 = no stacking)
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.frame_length_ms = frame_length_ms
        self.hop_length_ms = hop_length_ms
        self.use_deltas = use_deltas
        self.use_delta_deltas = use_delta_deltas
        self.context_frames = context_frames
        
        # Convert milliseconds to samples
        self.frame_length = int(sr * frame_length_ms / 1000)
        self.hop_length = int(sr * hop_length_ms / 1000)
        
        # Calculate expected feature dimensions
        self._calculate_feature_dims()
    
    def _calculate_feature_dims(self):
        """Calculate the total feature dimension."""
        # Base features per frame
        base_dim = self.n_mfcc
        
        # Add deltas
        if self.use_deltas:
            base_dim += self.n_mfcc
        if self.use_delta_deltas:
            base_dim += self.n_mfcc
        
        # Multiply by context window
        if self.context_frames > 0:
            context_size = 2 * self.context_frames + 1  # Before + current + after
            base_dim *= context_size
        
        self.feature_dim = base_dim
    
    def extract_frame_features(self, audio_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract frame-based features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (feature_matrix, metadata)
            - feature_matrix: Shape (n_frames, n_features) - each row is one time frame
            - metadata: Dictionary with extraction info
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Extract MFCCs frame by frame
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )  # Shape: (n_mfcc, n_frames)
        
        # Transpose to (n_frames, n_mfcc)
        mfccs = mfccs.T
        
        feature_list = [mfccs]
        
        # Compute delta features (first derivative)
        if self.use_deltas:
            mfcc_delta = librosa.feature.delta(mfccs.T).T
            feature_list.append(mfcc_delta)
        
        # Compute delta-delta features (second derivative)
        if self.use_delta_deltas:
            mfcc_delta2 = librosa.feature.delta(mfccs.T, order=2).T
            feature_list.append(mfcc_delta2)
        
        # Concatenate all features
        features = np.hstack(feature_list)  # Shape: (n_frames, n_features)
        
        # Apply context window (frame stacking) if requested
        if self.context_frames > 0:
            features = self._add_context_frames(features)
        
        metadata = {
            'n_frames': features.shape[0],
            'frame_length_ms': self.frame_length_ms,
            'hop_length_ms': self.hop_length_ms,
            'frame_length_samples': self.frame_length,
            'hop_length_samples': self.hop_length,
            'feature_dim': features.shape[1],
            'audio_duration_sec': len(y) / sr,
            'has_deltas': self.use_deltas,
            'has_delta_deltas': self.use_delta_deltas,
            'context_frames': self.context_frames
        }
        
        return features, metadata
    
    def _add_context_frames(self, features: np.ndarray) -> np.ndarray:
        """
        Stack consecutive frames to add temporal context.
        
        For context_frames=1:
        Each frame becomes [frame_{t-1}, frame_t, frame_{t+1}]
        
        Args:
            features: Shape (n_frames, n_features)
            
        Returns:
            Stacked features: Shape (n_frames, n_features * (2*context_frames + 1))
        """
        n_frames, n_features = features.shape
        context_size = 2 * self.context_frames + 1
        
        # Pad with edge values
        padded = np.pad(
            features, 
            ((self.context_frames, self.context_frames), (0, 0)),
            mode='edge'
        )
        
        # Stack frames
        stacked = []
        for i in range(n_frames):
            # Extract context window
            start = i
            end = i + context_size
            window = padded[start:end, :]  # Shape: (context_size, n_features)
            # Flatten to single vector
            stacked.append(window.flatten())
        
        return np.array(stacked)
    
    def extract_aggregated_features(self, audio_path: str) -> np.ndarray:
        """
        Extract frame features then aggregate with statistics.
        This is useful when you need a fixed-size vector per file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Aggregated feature vector (mean and std over all frames)
        """
        features, metadata = self.extract_frame_features(audio_path)
        
        # Compute statistics over time dimension
        mean_features = np.mean(features, axis=0)
        std_features = np.std(features, axis=0)
        
        # Concatenate mean and std
        aggregated = np.concatenate([mean_features, std_features])
        
        return aggregated
    
    def get_feature_names(self) -> list:
        """
        Get descriptive names for all features.
        
        Returns:
            List of feature names
        """
        base_names = [f'mfcc_{i}' for i in range(self.n_mfcc)]
        
        feature_names = base_names.copy()
        
        if self.use_deltas:
            feature_names.extend([f'delta_mfcc_{i}' for i in range(self.n_mfcc)])
        
        if self.use_delta_deltas:
            feature_names.extend([f'delta2_mfcc_{i}' for i in range(self.n_mfcc)])
        
        if self.context_frames > 0:
            # With context, features are stacked
            original_names = feature_names.copy()
            feature_names = []
            for offset in range(-self.context_frames, self.context_frames + 1):
                for name in original_names:
                    feature_names.append(f'{name}_t{offset:+d}')
        
        return feature_names
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return {
            'sr': self.sr,
            'n_mfcc': self.n_mfcc,
            'frame_length_ms': self.frame_length_ms,
            'hop_length_ms': self.hop_length_ms,
            'frame_length_samples': self.frame_length,
            'hop_length_samples': self.hop_length,
            'use_deltas': self.use_deltas,
            'use_delta_deltas': self.use_delta_deltas,
            'context_frames': self.context_frames,
            'feature_dim': self.feature_dim
        }


class TemporalDataAggregator:
    """
    Aggregate temporal features into fixed-size vectors for traditional ML.
    
    Since traditional ML models (Random Forest, SVM, etc.) need fixed-size inputs,
    we need to aggregate the variable-length frame sequences.
    """
    
    @staticmethod
    def aggregate_statistics(features: np.ndarray) -> np.ndarray:
        """
        Aggregate using mean and std over time.
        
        Args:
            features: Shape (n_frames, n_features)
            
        Returns:
            Aggregated vector: Shape (2 * n_features,)
        """
        mean_features = np.mean(features, axis=0)
        std_features = np.std(features, axis=0)
        return np.concatenate([mean_features, std_features])
    
    @staticmethod
    def aggregate_percentiles(features: np.ndarray, percentiles=[10, 25, 50, 75, 90]) -> np.ndarray:
        """
        Aggregate using percentiles over time.
        
        Args:
            features: Shape (n_frames, n_features)
            percentiles: List of percentiles to compute
            
        Returns:
            Aggregated vector: Shape (len(percentiles) * n_features,)
        """
        percentile_features = []
        for p in percentiles:
            percentile_features.append(np.percentile(features, p, axis=0))
        return np.concatenate(percentile_features)
    
    @staticmethod
    def aggregate_combined(features: np.ndarray) -> np.ndarray:
        """
        Aggregate using multiple statistics: mean, std, min, max, median.
        
        Args:
            features: Shape (n_frames, n_features)
            
        Returns:
            Aggregated vector: Shape (5 * n_features,)
        """
        stats = [
            np.mean(features, axis=0),
            np.std(features, axis=0),
            np.min(features, axis=0),
            np.max(features, axis=0),
            np.median(features, axis=0)
        ]
        return np.concatenate(stats)

