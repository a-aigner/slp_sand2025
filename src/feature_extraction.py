"""
Feature extraction module for audio files using librosa.
Extracts various audio features like MFCCs, spectral features, etc.
"""

import librosa
import numpy as np
from typing import Dict, Any


class AudioFeatureExtractor:
    """Extract audio features from wav files."""
    
    def __init__(self, sr: int = 22050, n_mfcc: int = 13):
        """
        Initialize the feature extractor.
        
        Args:
            sr: Sample rate for audio loading
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def extract_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing extracted features
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Extract MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Combine all features into a single feature vector
        features = {
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'spectral_contrast_mean': np.mean(spectral_contrast, axis=1),
            'spectral_contrast_std': np.std(spectral_contrast, axis=1),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_std': np.std(chroma, axis=1),
        }
        
        return features
    
    def features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert feature dictionary to a flat numpy array.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Flat feature vector
        """
        feature_list = []
        
        # Flatten all features in a consistent order
        feature_list.extend(features['mfcc_mean'])
        feature_list.extend(features['mfcc_std'])
        feature_list.append(features['spectral_centroid_mean'])
        feature_list.append(features['spectral_centroid_std'])
        feature_list.append(features['spectral_rolloff_mean'])
        feature_list.append(features['spectral_rolloff_std'])
        feature_list.extend(features['spectral_contrast_mean'])
        feature_list.extend(features['spectral_contrast_std'])
        feature_list.append(features['zcr_mean'])
        feature_list.append(features['zcr_std'])
        feature_list.extend(features['chroma_mean'])
        feature_list.extend(features['chroma_std'])
        
        return np.array(feature_list)

