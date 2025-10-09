#!/usr/bin/env python3
"""
Command-line application for training audio classification models.

Usage examples:
    # Train on a single folder (all .wav files get the same label)
    python train.py --data-dir data/task1/training/phonationA --label phonationA
    
    # Train on multiple folders (each folder becomes a class)
    python train.py --data-dir data/task1/training --subdirs phonationA phonationE phonationI
    
    # Train with custom model type
    python train.py --data-dir data/task1/training --model-type svm
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.feature_extraction import AudioFeatureExtractor
from src.temporal_features import TemporalFeatureExtractor
from src.data_loader import AudioDataLoader
from src.trainer import AudioClassifier


def main():
    parser = argparse.ArgumentParser(
        description='Train audio classification models on .wav files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on a single folder
  python train.py --data-dir data/task1/training/phonationA --label phonationA
  
  # Train on multiple folders (multi-class classification)
  python train.py --data-dir data/task1/training --subdirs phonationA phonationE phonationI
  
  # Train with metadata (Age, Sex) to predict Class from Excel file
  python train.py --data-dir data/task1/training --use-metadata --excel-file data/task1/sand_task_1.xlsx
  
  # Use different models
  python train.py --data-dir data/task1/training --model-type svm
  python train.py --data-dir data/task1/training --model-type logistic_regression --use-metadata --excel-file data/task1/sand_task_1.xlsx
        """
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing audio files or subdirectories with audio files'
    )
    parser.add_argument(
        '--subdirs',
        type=str,
        nargs='+',
        default=None,
        help='List of subdirectories to use (if not specified, uses all subdirectories)'
    )
    parser.add_argument(
        '--label',
        type=str,
        default=None,
        help='Label for single-directory mode (if specified, treats data-dir as containing files, not subdirs)'
    )
    parser.add_argument(
        '--use-metadata',
        action='store_true',
        help='Use metadata (Age, Sex) from Excel file as additional features to predict Class'
    )
    parser.add_argument(
        '--excel-file',
        type=str,
        default=None,
        help='Path to Excel file containing metadata (ID, Age, Sex, Class). Required if --use-metadata is set'
    )
    parser.add_argument(
        '--training-sheet',
        type=str,
        default='Training Baseline - Task 1',
        help='Name of Excel sheet with training data (default: "Training Baseline - Task 1")'
    )
    parser.add_argument(
        '--validation-sheet',
        type=str,
        default='Validation Baseline - Task 1',
        help='Name of Excel sheet with validation data (default: "Validation Baseline - Task 1")'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['random_forest', 'svm', 'logistic_regression', 'linear_regression'],
        default='random_forest',
        help='Type of model to train (default: random_forest)'
    )
    
    # Feature extraction arguments
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Sample rate for audio loading (default: 22050)'
    )
    parser.add_argument(
        '--n-mfcc',
        type=int,
        default=13,
        help='Number of MFCC coefficients (default: 13)'
    )
    
    # Temporal feature arguments
    parser.add_argument(
        '--use-temporal',
        action='store_true',
        help='Use temporal (time-series) features with frame-based processing'
    )
    parser.add_argument(
        '--frame-length',
        type=float,
        default=25.0,
        help='Frame length in milliseconds (default: 25.0ms, recommended: 10-25ms)'
    )
    parser.add_argument(
        '--hop-length',
        type=float,
        default=10.0,
        help='Hop length in milliseconds (default: 10.0ms)'
    )
    parser.add_argument(
        '--use-deltas',
        action='store_true',
        default=True,
        help='Include delta (velocity) features (default: True)'
    )
    parser.add_argument(
        '--use-delta-deltas',
        action='store_true',
        default=True,
        help='Include delta-delta (acceleration) features (default: True)'
    )
    parser.add_argument(
        '--context-frames',
        type=int,
        default=0,
        help='Number of frames before/after to stack for temporal context (default: 0, try: 1-3)'
    )
    
    # Output arguments
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Custom name for the saved model (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save confusion matrix and feature importance plots'
    )
    
    # Other arguments
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_metadata and args.excel_file is None:
        print("Error: --excel-file is required when --use-metadata is set")
        sys.exit(1)
    
    if args.use_metadata and args.label is not None:
        print("Error: --use-metadata cannot be used with --label (single-directory mode)")
        sys.exit(1)
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist")
        sys.exit(1)
    
    # Validate Excel file if using metadata
    if args.use_metadata:
        excel_path = Path(args.excel_file)
        if not excel_path.exists():
            print(f"Error: Excel file '{excel_path}' does not exist")
            sys.exit(1)
    
    print("=" * 80)
    print("AUDIO CLASSIFICATION TRAINING")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Use metadata: {args.use_metadata}")
    if args.use_metadata:
        print(f"Excel file: {args.excel_file}")
    print(f"Use temporal features: {args.use_temporal}")
    if args.use_temporal:
        print(f"  Frame length: {args.frame_length}ms")
        print(f"  Hop length: {args.hop_length}ms")
        print(f"  Delta features: {args.use_deltas}")
        print(f"  Delta-delta features: {args.use_delta_deltas}")
        print(f"  Context frames: {args.context_frames}")
    print(f"Training: Using 100% of training data")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Number of MFCCs: {args.n_mfcc}")
    print("=" * 80)
    
    # Initialize components
    if args.use_temporal:
        # Use temporal (frame-based) feature extraction
        feature_extractor = TemporalFeatureExtractor(
            sr=args.sample_rate,
            n_mfcc=args.n_mfcc,
            frame_length_ms=args.frame_length,
            hop_length_ms=args.hop_length,
            use_deltas=args.use_deltas,
            use_delta_deltas=args.use_delta_deltas,
            context_frames=args.context_frames
        )
        print(f"\nTemporal feature extractor initialized:")
        print(f"  Feature dimension per frame: {feature_extractor.feature_dim}")
        print(f"  Will aggregate frames using mean+std for fixed-size vector")
    else:
        # Use standard (aggregate) feature extraction
        feature_extractor = AudioFeatureExtractor(sr=args.sample_rate, n_mfcc=args.n_mfcc)
    
    data_loader = AudioDataLoader(feature_extractor=feature_extractor)
    classifier = AudioClassifier(model_type=args.model_type, random_state=args.random_seed)
    
    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    try:
        if args.use_metadata:
            # Metadata mode: load with Age/Sex features to predict Class
            print("Loading TRAINING data with metadata (Age, Sex) to predict Class...")
            X_train, y_train, metadata_train = data_loader.load_with_metadata(
                str(data_dir),
                args.excel_file,
                subdirs=args.subdirs,
                sheet_name=args.training_sheet
            )
            
            print("\nLoading VALIDATION data with metadata...")
            X_val, y_val, metadata_val = data_loader.load_with_metadata(
                str(data_dir),
                args.excel_file,
                subdirs=args.subdirs,
                sheet_name=args.validation_sheet
            )
            
            # Combine for training (will separate later)
            X = X_train
            y = y_train
        elif args.label is not None:
            # Single directory mode
            print(f"Loading from single directory with label '{args.label}'...")
            X, y, file_paths = data_loader.load_from_directory(str(data_dir), label=args.label)
            # For single directory, we might want binary classification or just save features
            if len(set(y)) == 1:
                print("\nWarning: Only one class found. Consider using multi-directory mode for classification.")
                print("Proceeding anyway - you can still save the model for feature extraction.")
        else:
            # Multi-directory mode
            print("Loading from multiple directories...")
            X, y, metadata = data_loader.load_from_multiple_directories(
                str(data_dir), 
                subdirs=args.subdirs
            )
    except Exception as e:
        print(f"\nError loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have enough data
    if len(X) < 10:
        print(f"\nWarning: Only {len(X)} samples found. Consider using more data for better results.")
    
    # For classification models, check number of classes
    if args.model_type != 'linear_regression':
        unique_classes = len(set(y))
        if unique_classes < 2:
            print("\nError: Need at least 2 classes for classification.")
            if args.use_metadata:
                print("Check your Excel file - the 'Class' column should have at least 2 different values.")
            else:
                print("Please use --subdirs to specify multiple class directories.")
            sys.exit(1)
    
    # Train model
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    
    try:
        if args.use_metadata:
            results = classifier.train(X, y, X_val, y_val)
        else:
            results = classifier.train(X, y)
    except Exception as e:
        print(f"\nError training model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save plots if requested
    if args.save_plots:
        print("\n" + "=" * 80)
        print("SAVING PLOTS")
        print("=" * 80)
        
        plots_dir = Path(args.model_dir) / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        cm_path = plots_dir / f'confusion_matrix_{args.model_type}.png'
        classifier.plot_confusion_matrix(results['confusion_matrix'], save_path=str(cm_path))
        
        # Feature importance (only for random forest)
        if args.model_type == 'random_forest':
            fi_path = plots_dir / f'feature_importance_{args.model_type}.png'
            classifier.plot_feature_importance(save_path=str(fi_path))
    
    # Save model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    model_path = classifier.save_model(args.model_dir, model_name=args.model_name)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    
    if args.model_type == 'linear_regression':
        # Regression summary
        print(f"Model type: Linear Regression")
        print(f"\nTraining R²: {results.get('train_r2', 'N/A'):.4f}")
        print(f"Training MSE: {results.get('train_mse', 'N/A'):.4f}")
        print(f"Training Accuracy (rounded): {results['train_accuracy']:.4f}")
        print(f"Cross-validation R²: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
    else:
        # Classification summary
        unique_classes = len(set(y))
        print(f"Number of classes: {unique_classes}")
        print(f"Classes: {', '.join(map(str, classifier.class_names))}")
        print(f"\nTraining Accuracy: {results['train_accuracy']:.4f}")
        print(f"Cross-validation Accuracy: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        if 'val_accuracy' in results:
            print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
    
    print(f"\nModel saved to: {model_path}")
    print("=" * 80)
    print("\nTraining complete! 🎉")
    print("\nNext steps:")
    print(f"  - Review the model performance metrics above")
    if args.save_plots:
        print(f"  - Check the plots in {plots_dir}")
    print(f"  - Use the saved model at {model_path} for predictions")
    if args.use_metadata:
        print(f"  - This model uses audio features + Age + Sex to predict Class")
    print("=" * 80)


if __name__ == '__main__':
    main()

