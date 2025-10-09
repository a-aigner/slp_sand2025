"""
Data loading module for audio classification.
Handles loading audio files from directories and extracting features.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from .feature_extraction import AudioFeatureExtractor
from .temporal_features import TemporalFeatureExtractor
from .metadata_loader import MetadataLoader


class AudioDataLoader:
    """Load and process audio files from directories."""
    
    def __init__(self, feature_extractor = None):
        """
        Initialize the data loader.
        
        Args:
            feature_extractor: AudioFeatureExtractor or TemporalFeatureExtractor instance
        """
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.is_temporal = isinstance(self.feature_extractor, TemporalFeatureExtractor)
    
    def load_from_directory(self, data_dir: str, label: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all .wav files from a directory and extract features.
        
        Args:
            data_dir: Directory containing .wav files
            label: Label for all files in this directory (optional)
            
        Returns:
            Tuple of (features array, labels array, file paths)
        """
        data_dir = Path(data_dir)
        
        # Get all .wav files
        wav_files = sorted(list(data_dir.glob('*.wav')))
        
        if not wav_files:
            raise ValueError(f"No .wav files found in {data_dir}")
        
        print(f"Found {len(wav_files)} .wav files in {data_dir}")
        
        # Extract features from each file
        features_list = []
        labels_list = []
        file_paths = []
        
        for wav_file in tqdm(wav_files, desc="Extracting features"):
            try:
                # Extract features (handles both temporal and standard extractors)
                if self.is_temporal:
                    # Temporal extractor returns aggregated features
                    feature_vector = self.feature_extractor.extract_aggregated_features(str(wav_file))
                else:
                    # Standard extractor
                    features_dict = self.feature_extractor.extract_features(str(wav_file))
                    feature_vector = self.feature_extractor.features_to_vector(features_dict)
                
                features_list.append(feature_vector)
                file_paths.append(str(wav_file))
                
                # Use provided label or directory name as label
                if label is not None:
                    labels_list.append(label)
                else:
                    labels_list.append(data_dir.name)
                    
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
                continue
        
        if not features_list:
            raise ValueError(f"Failed to extract features from any files in {data_dir}")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        return X, y, file_paths
    
    def load_from_multiple_directories(self, parent_dir: str, subdirs: List[str] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load audio files from multiple subdirectories, using subdirectory names as labels.
        
        Args:
            parent_dir: Parent directory containing subdirectories
            subdirs: List of subdirectory names to load (if None, loads all subdirectories)
            
        Returns:
            Tuple of (features array, labels array, metadata DataFrame)
        """
        parent_dir = Path(parent_dir)
        
        # Get subdirectories
        if subdirs is None:
            subdirs = [d.name for d in parent_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            raise ValueError(f"No subdirectories found in {parent_dir}")
        
        print(f"Loading data from {len(subdirs)} subdirectories: {subdirs}")
        
        all_features = []
        all_labels = []
        all_metadata = []
        
        for subdir in subdirs:
            subdir_path = parent_dir / subdir
            
            if not subdir_path.exists():
                print(f"Warning: {subdir_path} does not exist, skipping...")
                continue
            
            # Load data from this subdirectory
            X, y, file_paths = self.load_from_directory(str(subdir_path), label=subdir)
            
            all_features.append(X)
            all_labels.append(y)
            
            # Create metadata
            for fp in file_paths:
                all_metadata.append({
                    'file_path': fp,
                    'label': subdir,
                    'filename': Path(fp).name
                })
        
        # Combine all data
        X_combined = np.vstack(all_features)
        y_combined = np.concatenate(all_labels)
        metadata_df = pd.DataFrame(all_metadata)
        
        print(f"\nTotal samples loaded: {len(X_combined)}")
        print(f"Feature vector size: {X_combined.shape[1]}")
        print(f"Class distribution:\n{pd.Series(y_combined).value_counts()}")
        
        return X_combined, y_combined, metadata_df
    
    def load_with_metadata(
        self, 
        data_dir: str, 
        excel_path: str,
        subdirs: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load audio files and merge with metadata from Excel file.
        Uses Age and Sex as additional features, and Class as the target variable.
        
        Args:
            data_dir: Directory containing subdirectories with .wav files
            excel_path: Path to Excel file with metadata (ID, Age, Sex, Class)
            subdirs: List of subdirectory names to load (if None, loads all subdirectories)
            
        Returns:
            Tuple of (features array with metadata, class labels, metadata DataFrame)
        """
        print("=" * 80)
        print("LOADING DATA WITH METADATA")
        print("=" * 80)
        
        # Load metadata from Excel
        metadata_loader = MetadataLoader(excel_path)
        metadata_loader.load_metadata()
        
        # Display metadata summary
        summary = metadata_loader.get_metadata_summary()
        print(f"\nMetadata Summary:")
        print(f"  Total records: {summary['total_records']}")
        print(f"  Age range: {summary['age_stats']['min']:.0f} - {summary['age_stats']['max']:.0f}")
        print(f"  Age mean: {summary['age_stats']['mean']:.1f} ± {summary['age_stats']['std']:.1f}")
        print(f"  Sex distribution: {summary['sex_distribution']}")
        print(f"  Class distribution: {summary['class_distribution']}")
        
        # Get all audio files from subdirectories
        data_dir = Path(data_dir)
        
        if subdirs is None:
            subdirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            raise ValueError(f"No subdirectories found in {data_dir}")
        
        print(f"\nLoading audio from {len(subdirs)} subdirectories: {subdirs}")
        
        all_audio_features = []
        all_metadata_features = []
        all_classes = []
        all_file_info = []
        
        files_processed = 0
        files_without_metadata = 0
        
        for subdir in subdirs:
            subdir_path = data_dir / subdir
            
            if not subdir_path.exists():
                print(f"Warning: {subdir_path} does not exist, skipping...")
                continue
            
            # Get all .wav files
            wav_files = sorted(list(subdir_path.glob('*.wav')))
            
            if not wav_files:
                print(f"Warning: No .wav files found in {subdir_path}")
                continue
            
            print(f"\nProcessing {subdir}: {len(wav_files)} files")
            
            for wav_file in tqdm(wav_files, desc=f"Extracting {subdir}"):
                try:
                    # Extract audio features (handles both temporal and standard)
                    if self.is_temporal:
                        audio_feature_vector = self.feature_extractor.extract_aggregated_features(str(wav_file))
                    else:
                        features_dict = self.feature_extractor.extract_features(str(wav_file))
                        audio_feature_vector = self.feature_extractor.features_to_vector(features_dict)
                    
                    # Get metadata for this file
                    file_metadata = metadata_loader.get_metadata_for_file(str(wav_file))
                    
                    if file_metadata is None:
                        files_without_metadata += 1
                        # print(f"Warning: No metadata found for {wav_file.name}, skipping...")
                        continue
                    
                    # Extract metadata features (age, sex)
                    metadata_features = metadata_loader.prepare_metadata_features(file_metadata)
                    
                    # Store results
                    all_audio_features.append(audio_feature_vector)
                    all_metadata_features.append(metadata_features)
                    all_classes.append(file_metadata['class'])
                    
                    all_file_info.append({
                        'file_path': str(wav_file),
                        'filename': wav_file.name,
                        'subdirectory': subdir,
                        'id': file_metadata['id'],
                        'age': file_metadata['age'],
                        'sex': file_metadata['sex'],
                        'class': file_metadata['class']
                    })
                    
                    files_processed += 1
                    
                except Exception as e:
                    print(f"Error processing {wav_file}: {e}")
                    continue
        
        if not all_audio_features:
            raise ValueError("No files were successfully processed with metadata")
        
        # Combine audio features and metadata features
        audio_features_array = np.array(all_audio_features)
        metadata_features_array = np.array(all_metadata_features)
        
        # Concatenate: [audio_features, age, sex]
        X_combined = np.hstack([audio_features_array, metadata_features_array])
        y_classes = np.array(all_classes)
        metadata_df = pd.DataFrame(all_file_info)
        
        print("\n" + "=" * 80)
        print("LOADING SUMMARY")
        print("=" * 80)
        print(f"Files successfully processed: {files_processed}")
        print(f"Files without metadata: {files_without_metadata}")
        print(f"Total samples: {len(X_combined)}")
        print(f"Audio features: {audio_features_array.shape[1]}")
        print(f"Metadata features: {metadata_features_array.shape[1]} (age, sex)")
        print(f"Total features: {X_combined.shape[1]}")
        print(f"\nClass distribution:")
        print(pd.Series(y_classes).value_counts().sort_index())
        print("=" * 80)
        
        return X_combined, y_classes, metadata_df

