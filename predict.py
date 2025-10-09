#!/usr/bin/env python3
"""
Command-line application for making predictions on audio files using trained models.

Usage examples:
    # Predict on a single file
    python predict.py --model models/random_forest_20251009_143022.pkl --audio sample.wav
    
    # Predict on all files in a directory
    python predict.py --model models/random_forest_20251009_143022.pkl --audio-dir data/test/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.feature_extraction import AudioFeatureExtractor
from src.trainer import AudioClassifier


def predict_single_file(model_path: str, audio_path: str, sample_rate: int = 22050, n_mfcc: int = 13):
    """Predict the class of a single audio file."""
    # Load model
    classifier = AudioClassifier()
    classifier.load_model(model_path)
    
    # Extract features
    extractor = AudioFeatureExtractor(sr=sample_rate, n_mfcc=n_mfcc)
    features = extractor.extract_features(audio_path)
    feature_vector = extractor.features_to_vector(features)
    
    # Predict
    label, probabilities = classifier.predict([feature_vector])
    
    return label[0], probabilities[0] if probabilities is not None else None


def predict_directory(model_path: str, audio_dir: str, sample_rate: int = 22050, n_mfcc: int = 13):
    """Predict the class of all .wav files in a directory."""
    audio_dir = Path(audio_dir)
    wav_files = sorted(list(audio_dir.glob('*.wav')))
    
    if not wav_files:
        print(f"No .wav files found in {audio_dir}")
        return None
    
    # Load model
    classifier = AudioClassifier()
    classifier.load_model(model_path)
    
    # Extract features from all files
    extractor = AudioFeatureExtractor(sr=sample_rate, n_mfcc=n_mfcc)
    
    results = []
    print(f"\nProcessing {len(wav_files)} files...")
    
    for wav_file in wav_files:
        try:
            features = extractor.extract_features(str(wav_file))
            feature_vector = extractor.features_to_vector(features)
            
            # Predict
            label, probabilities = classifier.predict([feature_vector])
            
            result = {
                'filename': wav_file.name,
                'predicted_class': label[0]
            }
            
            # Add probabilities if available
            if probabilities is not None:
                for i, class_name in enumerate(classifier.class_names):
                    result[f'prob_{class_name}'] = probabilities[0][i]
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Make predictions on audio files using a trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on a single file
  python predict.py --model models/random_forest_20251009_143022.pkl --audio sample.wav
  
  # Predict on all files in a directory
  python predict.py --model models/random_forest_20251009_143022.pkl --audio-dir data/test/
  
  # Save predictions to CSV
  python predict.py --model models/my_model.pkl --audio-dir data/test/ --output predictions.csv
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model (.pkl file)'
    )
    
    # Input arguments (one of these is required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--audio',
        type=str,
        help='Path to a single audio file'
    )
    input_group.add_argument(
        '--audio-dir',
        type=str,
        help='Directory containing audio files'
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
    
    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for batch predictions (only for --audio-dir mode)'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model).exists():
        print(f"Error: Model file '{args.model}' does not exist")
        sys.exit(1)
    
    print("=" * 80)
    print("AUDIO CLASSIFICATION PREDICTION")
    print("=" * 80)
    print(f"Model: {args.model}")
    print("=" * 80)
    
    try:
        if args.audio:
            # Single file prediction
            if not Path(args.audio).exists():
                print(f"Error: Audio file '{args.audio}' does not exist")
                sys.exit(1)
            
            print(f"\nPredicting: {args.audio}")
            label, probabilities = predict_single_file(
                args.model, args.audio, 
                sample_rate=args.sample_rate, 
                n_mfcc=args.n_mfcc
            )
            
            print("\n" + "=" * 80)
            print("PREDICTION RESULT")
            print("=" * 80)
            print(f"Predicted Class: {label}")
            
            if probabilities is not None:
                print("\nClass Probabilities:")
                # Load model to get class names
                classifier = AudioClassifier()
                classifier.load_model(args.model)
                for i, class_name in enumerate(classifier.class_names):
                    print(f"  {class_name}: {probabilities[i]:.4f}")
            
            print("=" * 80)
            
        else:
            # Directory prediction
            if not Path(args.audio_dir).exists():
                print(f"Error: Directory '{args.audio_dir}' does not exist")
                sys.exit(1)
            
            print(f"\nPredicting on files in: {args.audio_dir}")
            results_df = predict_directory(
                args.model, args.audio_dir,
                sample_rate=args.sample_rate,
                n_mfcc=args.n_mfcc
            )
            
            if results_df is None or len(results_df) == 0:
                print("No predictions made.")
                sys.exit(1)
            
            print("\n" + "=" * 80)
            print("PREDICTION RESULTS")
            print("=" * 80)
            print(f"\nTotal files processed: {len(results_df)}")
            print("\nClass distribution:")
            print(results_df['predicted_class'].value_counts())
            
            # Save to CSV if requested
            if args.output:
                results_df.to_csv(args.output, index=False)
                print(f"\nResults saved to: {args.output}")
            else:
                print("\nFirst 10 predictions:")
                print(results_df.head(10).to_string(index=False))
            
            print("=" * 80)
    
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

