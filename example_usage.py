"""
Example script showing how to use the audio classification modules programmatically.

This demonstrates the low-level API for more advanced usage.
Most users should use train.py and predict.py instead.
"""

from pathlib import Path
from src.feature_extraction import AudioFeatureExtractor
from src.data_loader import AudioDataLoader
from src.trainer import AudioClassifier


def example_1_basic_training():
    """Example 1: Basic training workflow."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Training Workflow")
    print("=" * 80)
    
    # Initialize components
    feature_extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13)
    data_loader = AudioDataLoader(feature_extractor=feature_extractor)
    
    # Load data from multiple directories
    X, y, metadata = data_loader.load_from_multiple_directories(
        parent_dir='data/task1/training',
        subdirs=['phonationA', 'phonationE', 'phonationI']
    )
    
    print(f"\nLoaded {len(X)} samples with {X.shape[1]} features")
    print(f"Classes: {set(y)}")
    
    # Train classifier
    classifier = AudioClassifier(model_type='random_forest', random_state=42)
    results = classifier.train(X, y, test_size=0.2)
    
    print(f"\nTest Accuracy: {results['test_accuracy']:.4f}")
    
    # Save model
    model_path = classifier.save_model('models', model_name='example_model')
    
    return classifier, results


def example_2_feature_extraction():
    """Example 2: Extract features from a single audio file."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Feature Extraction")
    print("=" * 80)
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13)
    
    # Get first .wav file as example
    audio_file = next(Path('data/task1/training/phonationA').glob('*.wav'))
    print(f"\nExtracting features from: {audio_file.name}")
    
    # Extract features
    features = extractor.extract_features(str(audio_file))
    
    # Show some features
    print(f"\nMFCC means: {features['mfcc_mean'][:5]}... (showing first 5)")
    print(f"Spectral centroid mean: {features['spectral_centroid_mean']:.2f}")
    print(f"Zero crossing rate mean: {features['zcr_mean']:.6f}")
    
    # Convert to feature vector
    feature_vector = extractor.features_to_vector(features)
    print(f"\nFeature vector length: {len(feature_vector)}")
    
    return features, feature_vector


def example_3_load_and_predict():
    """Example 3: Load a saved model and make predictions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Load Model and Predict")
    print("=" * 80)
    
    # Check if we have a saved model
    model_path = Path('models/example_model.pkl')
    if not model_path.exists():
        print("\nNo saved model found. Run example_1_basic_training() first.")
        return
    
    # Load model
    classifier = AudioClassifier()
    classifier.load_model(str(model_path))
    
    # Extract features from a new file
    extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13)
    audio_file = next(Path('data/task1/training/phonationE').glob('*.wav'))
    
    print(f"\nPredicting on: {audio_file.name}")
    
    features = extractor.extract_features(str(audio_file))
    feature_vector = extractor.features_to_vector(features)
    
    # Make prediction
    predicted_label, probabilities = classifier.predict([feature_vector])
    
    print(f"\nPredicted class: {predicted_label[0]}")
    
    if probabilities is not None:
        print("\nClass probabilities:")
        for i, class_name in enumerate(classifier.class_names):
            print(f"  {class_name}: {probabilities[0][i]:.4f}")
    
    return predicted_label, probabilities


def example_4_custom_pipeline():
    """Example 4: Create a custom pipeline with different parameters."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Custom Pipeline")
    print("=" * 80)
    
    # Use more MFCC coefficients
    feature_extractor = AudioFeatureExtractor(sr=22050, n_mfcc=20)
    data_loader = AudioDataLoader(feature_extractor=feature_extractor)
    
    # Load only rhythm data
    X, y, metadata = data_loader.load_from_multiple_directories(
        parent_dir='data/task1/training',
        subdirs=['rhythmKA', 'rhythmPA', 'rhythmTA']
    )
    
    print(f"\nFeature vector size: {X.shape[1]}")
    
    # Train SVM with larger test set
    classifier = AudioClassifier(model_type='svm', random_state=42)
    results = classifier.train(X, y, test_size=0.3)
    
    print(f"\nSVM Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Cross-validation: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
    
    return classifier, results


def example_5_single_directory():
    """Example 5: Load from a single directory."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Single Directory Loading")
    print("=" * 80)
    
    # Load from a single directory
    feature_extractor = AudioFeatureExtractor()
    data_loader = AudioDataLoader(feature_extractor=feature_extractor)
    
    X, y, file_paths = data_loader.load_from_directory(
        data_dir='data/task1/training/phonationA',
        label='phonationA'
    )
    
    print(f"\nLoaded {len(X)} samples")
    print(f"All labels: {set(y)}")
    print(f"First 3 files:")
    for i, fp in enumerate(file_paths[:3]):
        print(f"  {i+1}. {Path(fp).name}")
    
    return X, y, file_paths


if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "AUDIO CLASSIFICATION EXAMPLES" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Check if data exists
    if not Path('data/task1/training/phonationA').exists():
        print("\nError: Data directory not found.")
        print("Make sure you have data in data/task1/training/")
        exit(1)
    
    try:
        # Run examples
        print("\nRunning examples... This may take a few minutes.\n")
        
        # Example 1: Basic training
        classifier1, results1 = example_1_basic_training()
        
        # Example 2: Feature extraction
        features, feature_vector = example_2_feature_extraction()
        
        # Example 3: Load and predict
        example_3_load_and_predict()
        
        # Example 4: Custom pipeline
        classifier4, results4 = example_4_custom_pipeline()
        
        # Example 5: Single directory
        X5, y5, paths5 = example_5_single_directory()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✅")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. Use AudioFeatureExtractor to extract features from .wav files")
        print("  2. Use AudioDataLoader to load data from directories")
        print("  3. Use AudioClassifier to train and evaluate models")
        print("  4. Save and load models with classifier.save_model() / load_model()")
        print("  5. Make predictions with classifier.predict()")
        print("\nFor production use, prefer the CLI tools: train.py and predict.py")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

