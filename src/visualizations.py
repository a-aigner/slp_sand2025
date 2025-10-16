#!/usr/bin/env python3
"""
Comprehensive visualization helper module for audio classification analysis.

This module provides extensive visualization functions for:
- Audio analysis (waveforms, spectrograms, MFCCs, mel spectrograms)
- Feature analysis (distributions, correlations, PCA, t-SNE)
- Model performance (confusion matrices, ROC curves, precision-recall, learning curves)
- Data exploration (class distributions, metadata, imbalance)
- Comparative analysis (model comparison, feature importance)
- Statistical analysis (violin plots, box plots, kde plots)

Author: Audio ML Visualization Suite
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Import ML/analysis libraries
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        confusion_matrix, classification_report, 
        roc_curve, auc, precision_recall_curve,
        roc_auc_score, average_precision_score
    )
    from sklearn.model_selection import learning_curve
    import scikitplot as skplt
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some visualizations will be limited.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive plots will be disabled.")


class AudioVisualizer:
    """
    Comprehensive audio visualization class with static methods for various plot types.
    All methods support saving to file and customization via parameters.
    """
    
    # ==================== AUDIO ANALYSIS ====================
    
    @staticmethod
    def plot_waveform(
        audio: np.ndarray,
        sr: int = 22050,
        title: str = "Waveform",
        figsize: Tuple[int, int] = (14, 4),
        color: str = '#1f77b4',
        alpha: float = 0.7,
        show_grid: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot audio waveform with time axis.
        
        Args:
            audio: Audio time series
            sr: Sample rate
            title: Plot title
            figsize: Figure size (width, height)
            color: Line color
            alpha: Line transparency
            show_grid: Whether to show grid
            save_path: Path to save figure (if None, not saved)
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create time axis
        time = np.arange(0, len(audio)) / sr
        
        ax.plot(time, audio, color=color, alpha=alpha, linewidth=0.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_spectrogram(
        audio: np.ndarray,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        title: str = "Spectrogram",
        figsize: Tuple[int, int] = (14, 6),
        cmap: str = 'viridis',
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot spectrogram of audio signal.
        
        Args:
            audio: Audio time series
            sr: Sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            title: Plot title
            figsize: Figure size
            cmap: Colormap
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute spectrogram
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        
        img = librosa.display.specshow(
            D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz',
            cmap=cmap, ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_mel_spectrogram(
        audio: np.ndarray,
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        title: str = "Mel Spectrogram",
        figsize: Tuple[int, int] = (14, 6),
        cmap: str = 'magma',
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot mel spectrogram of audio signal.
        
        Args:
            audio: Audio time series
            sr: Sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length
            title: Plot title
            figsize: Figure size
            cmap: Colormap
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        img = librosa.display.specshow(
            mel_spec_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='mel', cmap=cmap, ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Mel Frequency', fontsize=12)
        
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_mfcc(
        audio: np.ndarray,
        sr: int = 22050,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        title: str = "MFCC",
        figsize: Tuple[int, int] = (14, 6),
        cmap: str = 'coolwarm',
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot MFCC coefficients over time.
        
        Args:
            audio: Audio time series
            sr: Sample rate
            n_mfcc: Number of MFCCs to extract
            n_fft: FFT window size
            hop_length: Hop length
            title: Plot title
            figsize: Figure size
            cmap: Colormap
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
        
        img = librosa.display.specshow(
            mfccs, sr=sr, hop_length=hop_length, x_axis='time', cmap=cmap, ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('MFCC Coefficients', fontsize=12)
        
        fig.colorbar(img, ax=ax)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_audio_comparison(
        audio_dict: Dict[str, Tuple[np.ndarray, int]],
        plot_type: str = 'waveform',
        figsize: Tuple[int, int] = (14, 10),
        cmap: str = 'viridis',
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Compare multiple audio files side by side.
        
        Args:
            audio_dict: Dictionary mapping labels to (audio, sr) tuples
            plot_type: Type of plot ('waveform', 'spectrogram', 'mel', 'mfcc')
            figsize: Figure size
            cmap: Colormap for spectral plots
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        n_audio = len(audio_dict)
        fig, axes = plt.subplots(n_audio, 1, figsize=figsize)
        
        if n_audio == 1:
            axes = [axes]
        
        for idx, (label, (audio, sr)) in enumerate(audio_dict.items()):
            ax = axes[idx]
            
            if plot_type == 'waveform':
                time = np.arange(0, len(audio)) / sr
                ax.plot(time, audio, linewidth=0.5)
                ax.set_ylabel('Amplitude')
                if idx == n_audio - 1:
                    ax.set_xlabel('Time (s)')
            
            elif plot_type == 'spectrogram':
                D = librosa.amplitude_to_db(
                    np.abs(librosa.stft(audio)), ref=np.max
                )
                librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                        cmap=cmap, ax=ax)
                ax.set_ylabel('Frequency (Hz)')
            
            elif plot_type == 'mel':
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', 
                                        y_axis='mel', cmap=cmap, ax=ax)
                ax.set_ylabel('Mel Freq')
            
            elif plot_type == 'mfcc':
                mfccs = librosa.feature.mfcc(y=audio, sr=sr)
                librosa.display.specshow(mfccs, sr=sr, x_axis='time', 
                                        cmap=cmap, ax=ax)
                ax.set_ylabel('MFCC')
            
            ax.set_title(label, fontweight='bold')
        
        plt.suptitle(f'{plot_type.capitalize()} Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_chromagram(
        audio: np.ndarray,
        sr: int = 22050,
        hop_length: int = 512,
        title: str = "Chromagram",
        figsize: Tuple[int, int] = (14, 6),
        cmap: str = 'coolwarm',
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot chromagram showing pitch class distribution over time.
        
        Args:
            audio: Audio time series
            sr: Sample rate
            hop_length: Hop length
            title: Plot title
            figsize: Figure size
            cmap: Colormap
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)
        
        img = librosa.display.specshow(
            chroma, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma',
            cmap=cmap, ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Pitch Class', fontsize=12)
        
        fig.colorbar(img, ax=ax)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    # ==================== FEATURE ANALYSIS ====================
    
    @staticmethod
    def plot_feature_distributions(
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_cols: int = 4,
        figsize: Tuple[int, int] = (16, 12),
        color: str = 'skyblue',
        kde: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot histogram distributions of all features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            n_cols: Number of columns in subplot grid
            figsize: Figure size
            color: Histogram color
            kde: Whether to show KDE overlay
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        n_features = X.shape[1]
        n_rows = int(np.ceil(n_features / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        for idx in range(n_features):
            ax = axes[idx]
            sns.histplot(X[:, idx], bins=30, color=color, kde=kde, ax=ax)
            ax.set_title(feature_names[idx], fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('Count' if idx % n_cols == 0 else '')
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_feature_correlation(
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 12),
        cmap: str = 'coolwarm',
        annot: bool = False,
        vmin: float = -1,
        vmax: float = 1,
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot correlation heatmap of features.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            figsize: Figure size
            cmap: Colormap
            annot: Whether to annotate cells with values
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute correlation matrix
        if isinstance(X, pd.DataFrame):
            corr = X.corr()
        else:
            df = pd.DataFrame(X, columns=feature_names)
            corr = df.corr()
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr, mask=mask, cmap=cmap, center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, annot=annot,
            vmin=vmin, vmax=vmax, ax=ax, fmt='.2f'
        )
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_pca(
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n_components: int = 2,
        figsize: Tuple[int, int] = (12, 8),
        alpha: float = 0.6,
        s: int = 50,
        class_names: Optional[List[str]] = None,
        title: str = "PCA Projection",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot PCA projection of features.
        
        Args:
            X: Feature matrix
            y: Labels (optional, for coloring)
            n_components: Number of PCA components (2 or 3)
            figsize: Figure size
            alpha: Point transparency
            s: Point size
            class_names: Class names for legend
            title: Plot title
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for PCA")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        if n_components == 2:
            fig, ax = plt.subplots(figsize=figsize)
            
            if y is not None:
                unique_classes = np.unique(y)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
                
                for idx, cls in enumerate(unique_classes):
                    mask = y == cls
                    label = class_names[idx] if class_names else f'Class {cls}'
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                             c=[colors[idx]], label=label, alpha=alpha, s=s)
                ax.legend()
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=alpha, s=s)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', 
                         fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', 
                         fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        elif n_components == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            if y is not None:
                unique_classes = np.unique(y)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
                
                for idx, cls in enumerate(unique_classes):
                    mask = y == cls
                    label = class_names[idx] if class_names else f'Class {cls}'
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                             c=[colors[idx]], label=label, alpha=alpha, s=s)
                ax.legend()
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=alpha, s=s)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=10)
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_tsne(
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        perplexity: int = 30,
        n_iter: int = 1000,
        figsize: Tuple[int, int] = (12, 8),
        alpha: float = 0.6,
        s: int = 50,
        class_names: Optional[List[str]] = None,
        title: str = "t-SNE Projection",
        random_state: int = 42,
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot t-SNE projection of features.
        
        Args:
            X: Feature matrix
            y: Labels (optional, for coloring)
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations
            figsize: Figure size
            alpha: Point transparency
            s: Point size
            class_names: Class names for legend
            title: Plot title
            random_state: Random seed
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for t-SNE")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                   random_state=random_state)
        X_tsne = tsne.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if y is not None:
            unique_classes = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
            
            for idx, cls in enumerate(unique_classes):
                mask = y == cls
                label = class_names[idx] if class_names else f'Class {cls}'
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                         c=[colors[idx]], label=label, alpha=alpha, s=s)
            ax.legend()
        else:
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=alpha, s=s)
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_pca_variance(
        X: np.ndarray,
        max_components: int = 20,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot PCA explained variance ratio.
        
        Args:
            X: Feature matrix
            max_components: Maximum number of components to show
            figsize: Figure size
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for PCA")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_components = min(max_components, X.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Individual variance
        ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_, 
               color='steelblue', alpha=0.8)
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.set_title('Variance Explained by Each PC', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Cumulative variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(range(1, n_components + 1), cumsum, marker='o', 
                color='darkgreen', linewidth=2, markersize=6)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax2.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    # ==================== DATA EXPLORATION ====================
    
    @staticmethod
    def plot_class_distribution(
        y: np.ndarray,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        color: str = 'steelblue',
        title: str = "Class Distribution",
        show_percentages: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot class distribution as bar chart.
        
        Args:
            y: Class labels
            class_names: Names for each class
            figsize: Figure size
            color: Bar color
            title: Plot title
            show_percentages: Whether to show percentages on bars
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        unique, counts = np.unique(y, return_counts=True)
        
        if class_names is None:
            class_names = [f'Class {c}' for c in unique]
        
        bars = ax.bar(class_names, counts, color=color, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = count / len(y) * 100
            
            if show_percentages:
                label = f'{count}\n({percentage:.1f}%)'
            else:
                label = f'{count}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels if needed
        if len(class_names) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_metadata_analysis(
        df: pd.DataFrame,
        metadata_cols: List[str],
        target_col: str = 'Class',
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Analyze metadata features in relation to target variable.
        
        Args:
            df: DataFrame with metadata
            metadata_cols: List of metadata column names
            target_col: Target column name
            figsize: Figure size
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        n_cols = len(metadata_cols)
        n_rows = int(np.ceil(n_cols / 2))
        
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(metadata_cols):
            ax = axes[idx]
            
            if df[col].dtype in ['int64', 'float64']:
                # Numerical: box plot or violin plot
                if target_col in df.columns:
                    sns.violinplot(data=df, x=target_col, y=col, ax=ax, palette='Set2')
                    ax.set_title(f'{col} by {target_col}', fontweight='bold')
                else:
                    sns.histplot(data=df, x=col, bins=20, ax=ax, kde=True)
                    ax.set_title(f'{col} Distribution', fontweight='bold')
            else:
                # Categorical: count plot
                if target_col in df.columns:
                    sns.countplot(data=df, x=col, hue=target_col, ax=ax, palette='Set1')
                    ax.set_title(f'{col} by {target_col}', fontweight='bold')
                    ax.legend(title=target_col)
                else:
                    sns.countplot(data=df, x=col, ax=ax)
                    ax.set_title(f'{col} Distribution', fontweight='bold')
                
                if len(df[col].unique()) > 3:
                    ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Metadata Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_sample_counts_by_category(
        categories: List[str],
        counts: List[int],
        figsize: Tuple[int, int] = (12, 6),
        color_palette: str = 'viridis',
        title: str = "Sample Counts by Category",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot sample counts across different categories.
        
        Args:
            categories: List of category names
            counts: List of counts for each category
            figsize: Figure size
            color_palette: Color palette name
            title: Plot title
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = sns.color_palette(color_palette, len(categories))
        bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    # ==================== MODEL PERFORMANCE ====================
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = False,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'Blues',
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names
            normalize: Whether to normalize (show percentages)
            figsize: Figure size
            cmap: Colormap
            title: Plot title
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for confusion matrix")
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, square=True,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Percentage' if normalize else 'Count'},
                   ax=ax, linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_roc_curves(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        title: str = "ROC Curves",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels (one-hot or integer)
            y_pred_proba: Predicted probabilities (n_samples, n_classes)
            class_names: Class names
            figsize: Figure size
            title: Plot title
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for ROC curves")
        
        from sklearn.preprocessing import label_binarize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Binarize labels if needed
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        if y_true.ndim == 1:
            y_true_bin = label_binarize(y_true, classes=unique_classes)
            if n_classes == 2:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        else:
            y_true_bin = y_true
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            label = class_names[i] if class_names else f'Class {unique_classes[i]}'
            ax.plot(fpr, tpr, color=colors[i], lw=2, 
                   label=f'{label} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_precision_recall_curves(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        title: str = "Precision-Recall Curves",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot precision-recall curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            class_names: Class names
            figsize: Figure size
            title: Plot title
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for precision-recall curves")
        
        from sklearn.preprocessing import label_binarize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        if y_true.ndim == 1:
            y_true_bin = label_binarize(y_true, classes=unique_classes)
            if n_classes == 2:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        else:
            y_true_bin = y_true
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], 
                                                         y_pred_proba[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            
            label = class_names[i] if class_names else f'Class {unique_classes[i]}'
            ax.plot(recall, precision, color=colors[i], lw=2,
                   label=f'{label} (AP = {avg_precision:.2f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_feature_importance(
        importances: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        color: str = 'forestgreen',
        title: str = "Feature Importance",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot feature importance scores.
        
        Args:
            importances: Feature importance scores
            feature_names: Feature names
            top_n: Number of top features to show
            figsize: Figure size
            color: Bar color
            title: Plot title
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Plot
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importances[indices], color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_learning_curve(
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        n_jobs: int = -1,
        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Learning Curve",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot learning curve showing training and validation scores vs dataset size.
        
        Args:
            estimator: ML model
            X: Features
            y: Labels
            cv: Cross-validation folds
            n_jobs: Number of parallel jobs
            train_sizes: Training set sizes to evaluate
            figsize: Figure size
            title: Plot title
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for learning curves")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
            shuffle=True, random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='r')
        
        ax.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='g')
        
        ax.set_xlabel('Training Examples', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    # ==================== COMPARATIVE ANALYSIS ====================
    
    @staticmethod
    def plot_model_comparison(
        model_names: List[str],
        metrics_dict: Dict[str, List[float]],
        figsize: Tuple[int, int] = (12, 6),
        title: str = "Model Comparison",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Compare multiple models across different metrics.
        
        Args:
            model_names: List of model names
            metrics_dict: Dictionary mapping metric names to lists of values
            figsize: Figure size
            title: Plot title
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        n_metrics = len(metrics_dict)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        colors = sns.color_palette('Set2', len(model_names))
        
        for idx, (metric_name, values) in enumerate(metrics_dict.items()):
            ax = axes[idx]
            
            bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(metric_name, fontweight='bold')
            ax.set_ylim([0, max(values) * 1.15])
            ax.grid(True, alpha=0.3, axis='y')
            
            if len(model_names) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_grouped_comparison(
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        hue_col: str,
        plot_type: str = 'bar',
        figsize: Tuple[int, int] = (12, 6),
        title: str = "Grouped Comparison",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Create grouped comparison plot (bar, box, or violin).
        
        Args:
            data: DataFrame with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            hue_col: Column for grouping
            plot_type: Type of plot ('bar', 'box', 'violin')
            figsize: Figure size
            title: Plot title
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if plot_type == 'bar':
            sns.barplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax, palette='Set2')
        elif plot_type == 'box':
            sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax, palette='Set2')
        elif plot_type == 'violin':
            sns.violinplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax, palette='Set2')
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(title=hue_col)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        return fig
    
    # ==================== INTERACTIVE PLOTS (PLOTLY) ====================
    
    @staticmethod
    def plot_interactive_pca(
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        title: str = "Interactive PCA",
        save_path: Optional[str] = None
    ):
        """
        Create interactive 3D PCA plot using Plotly.
        
        Args:
            X: Feature matrix
            y: Labels (optional)
            class_names: Class names
            feature_names: Feature names
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for interactive plots")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for PCA")
        
        # Standardize and apply PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame
        df_plot = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'PC3': X_pca[:, 2]
        })
        
        if y is not None:
            if class_names is not None:
                df_plot['Class'] = [class_names[int(yi)] for yi in y]
            else:
                df_plot['Class'] = y.astype(str)
            color = 'Class'
        else:
            color = None
        
        fig = px.scatter_3d(
            df_plot, x='PC1', y='PC2', z='PC3', color=color,
            title=title,
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
            }
        )
        
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive plot to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_interactive_correlation(
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str = "Interactive Correlation Matrix",
        save_path: Optional[str] = None
    ):
        """
        Create interactive correlation heatmap using Plotly.
        
        Args:
            X: Feature matrix
            feature_names: Feature names
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for interactive plots")
        
        if isinstance(X, pd.DataFrame):
            corr = X.corr()
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            df = pd.DataFrame(X, columns=feature_names)
            corr = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=feature_names if feature_names else corr.columns,
            y=feature_names if feature_names else corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Features",
            width=900,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive plot to: {save_path}")
        
        return fig
    
    # ==================== UTILITY FUNCTIONS ====================
    
    @staticmethod
    def create_multi_panel_figure(
        n_rows: int,
        n_cols: int,
        figsize: Tuple[int, int] = (16, 12),
        title: str = "Multi-Panel Analysis"
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a multi-panel figure for combining multiple plots.
        
        Args:
            n_rows: Number of rows
            n_cols: Number of columns
            figsize: Figure size
            title: Overall title
            
        Returns:
            Tuple of (figure, axes array)
        """
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        plt.suptitle(title, fontsize=16, fontweight='bold')
        return fig, axes
    
    @staticmethod
    def save_multiple_formats(
        fig: plt.Figure,
        base_path: str,
        formats: List[str] = ['png', 'pdf', 'svg'],
        dpi: int = 300
    ):
        """
        Save figure in multiple formats.
        
        Args:
            fig: Matplotlib figure
            base_path: Base path without extension
            formats: List of formats to save
            dpi: DPI for raster formats
        """
        for fmt in formats:
            save_path = f"{base_path}.{fmt}"
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")


# ==================== CONVENIENCE FUNCTIONS ====================

def quick_audio_analysis(
    audio_path: str,
    sr: int = 22050,
    save_dir: Optional[str] = None,
    dpi: int = 300
) -> Dict[str, plt.Figure]:
    """
    Quickly generate all audio analysis plots for a single file.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        save_dir: Directory to save plots (if None, not saved)
        dpi: DPI for saved figures
        
    Returns:
        Dictionary of figure objects
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sr)
    filename = Path(audio_path).stem
    
    figures = {}
    viz = AudioVisualizer()
    
    # Waveform
    save_path = f"{save_dir}/{filename}_waveform.png" if save_dir else None
    figures['waveform'] = viz.plot_waveform(audio, sr, title=f"Waveform - {filename}", 
                                           save_path=save_path, dpi=dpi)
    
    # Spectrogram
    save_path = f"{save_dir}/{filename}_spectrogram.png" if save_dir else None
    figures['spectrogram'] = viz.plot_spectrogram(audio, sr, title=f"Spectrogram - {filename}",
                                                 save_path=save_path, dpi=dpi)
    
    # Mel Spectrogram
    save_path = f"{save_dir}/{filename}_mel.png" if save_dir else None
    figures['mel'] = viz.plot_mel_spectrogram(audio, sr, title=f"Mel Spectrogram - {filename}",
                                             save_path=save_path, dpi=dpi)
    
    # MFCC
    save_path = f"{save_dir}/{filename}_mfcc.png" if save_dir else None
    figures['mfcc'] = viz.plot_mfcc(audio, sr, title=f"MFCC - {filename}",
                                   save_path=save_path, dpi=dpi)
    
    # Chromagram
    save_path = f"{save_dir}/{filename}_chroma.png" if save_dir else None
    figures['chroma'] = viz.plot_chromagram(audio, sr, title=f"Chromagram - {filename}",
                                           save_path=save_path, dpi=dpi)
    
    print(f"Generated {len(figures)} plots for {filename}")
    return figures


# Example usage documentation
if __name__ == "__main__":
    print("""
    Audio Classification Visualization Module
    =========================================
    
    This module provides comprehensive visualization tools for audio classification analysis.
    
    Quick Examples:
    ---------------
    
    # Audio Analysis
    from src.visualizations import AudioVisualizer
    import librosa
    
    audio, sr = librosa.load('audio.wav')
    viz = AudioVisualizer()
    
    viz.plot_waveform(audio, sr, save_path='waveform.png')
    viz.plot_spectrogram(audio, sr, save_path='spectrogram.png')
    viz.plot_mfcc(audio, sr, save_path='mfcc.png')
    
    # Feature Analysis
    viz.plot_feature_distributions(X, feature_names=names, save_path='features.png')
    viz.plot_feature_correlation(X, feature_names=names, save_path='correlation.png')
    viz.plot_pca(X, y, class_names=classes, save_path='pca.png')
    viz.plot_tsne(X, y, class_names=classes, save_path='tsne.png')
    
    # Model Performance
    viz.plot_confusion_matrix(y_true, y_pred, class_names=classes, save_path='cm.png')
    viz.plot_roc_curves(y_true, y_pred_proba, class_names=classes, save_path='roc.png')
    viz.plot_feature_importance(importances, feature_names=names, save_path='importance.png')
    
    # Quick Analysis
    from src.visualizations import quick_audio_analysis
    quick_audio_analysis('audio.wav', save_dir='output/')
    
    See individual function docstrings for detailed parameter information.
    """)

