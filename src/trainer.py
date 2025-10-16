"""
Training module with sklearn pipelines.
Handles model training, evaluation, and saving.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class AudioClassifier:
    """Train and evaluate audio classification models."""
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of classifier ('random_forest', 'svm', 'logistic_regression', 'linear_regression')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        self.is_regression = model_type == 'linear_regression'
        
    def create_pipeline(self) -> Pipeline:
        """
        Create a sklearn pipeline with preprocessing and model.
        
        Returns:
            Configured sklearn Pipeline
        """
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            model = SVC(
                kernel='rbf',
                C=1.0,
                random_state=self.random_state,
                class_weight='balanced',
                probability=True
            )
        elif self.model_type == 'logistic_regression':
            model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                multi_class='multinomial',
                solver='lbfgs',
                class_weight='balanced'
            )
        elif self.model_type == 'linear_regression':
            model = LinearRegression(
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create pipeline with scaling and model
        # Use different step name for regression vs classification
        step_name = 'regressor' if self.is_regression else 'classifier'
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (step_name, model)
        ])
        
        return pipeline
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the model on all provided training data and evaluate on validation data if provided.
        
        Args:
            X: Training feature matrix
            y: Training labels (strings for classification) or numeric values (for regression)
            X_val: Validation feature matrix (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training results and metrics
        """
        print(f"\nTraining {self.model_type} model...")
        print(f"Using all {len(X)} samples for training")
        if X_val is not None:
            print(f"Validation set size: {len(X_val)}")
        
        # Handle regression vs classification
        if self.is_regression:
            # For regression, keep numeric values
            y_train = y.astype(float)
        else:
            # For classification, encode labels
            y_train = self.label_encoder.fit_transform(y)
            self.class_names = self.label_encoder.classes_
        
        X_train = X
        
        print(f"Training set size: {len(X_train)}")
        
        # Create and train pipeline
        self.pipeline = self.create_pipeline()
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate based on model type
        if self.is_regression:
            # Regression metrics on training data
            y_train_pred = self.pipeline.predict(X_train)
            
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            
            # Cross-validation for regression
            cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring='r2')
            
            print(f"\nTraining R²: {train_r2:.4f}, MSE: {train_mse:.4f}")
            print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Round predictions to nearest class for evaluation
            y_train_pred_rounded = np.round(y_train_pred).astype(int)
            y_train_rounded = np.round(y_train).astype(int)
            
            # Calculate accuracy when treating as classification
            rounded_accuracy = accuracy_score(y_train_rounded, y_train_pred_rounded)
            print(f"Training Accuracy (rounded to nearest class): {rounded_accuracy:.4f}")
            
            # Display confusion matrix for rounded predictions
            cm_rounded = confusion_matrix(y_train_rounded, y_train_pred_rounded)
            # Set class names for regression (numeric values)
            unique_classes = np.unique(np.concatenate([y_train_rounded, y_train_pred_rounded]))
            self.class_names = unique_classes
            self._print_confusion_matrix(cm_rounded)
            
            results = {
                'train_r2': train_r2,
                'train_mse': train_mse,
                'train_accuracy': rounded_accuracy,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': cm_rounded
            }
        else:
            # Classification metrics
            y_train_pred = self.pipeline.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5)
            
            print(f"\nTraining Accuracy: {train_accuracy:.4f}")
            print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            results = {
                'train_accuracy': train_accuracy,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
            }
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                # Encode validation labels
                y_val_encoded = self.label_encoder.transform(y_val)
                y_val_pred = self.pipeline.predict(X_val)
                val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
                
                print(f"\n{'='*80}")
                print("VALIDATION RESULTS")
                print(f"{'='*80}")
                print(f"Validation Accuracy: {val_accuracy:.4f}")
                
                # Generate classification report on validation data
                print("\nClassification Report (Validation Data):")
                class_names_str = [str(name) for name in self.class_names]
                report = classification_report(
                    y_val_encoded, y_val_pred, 
                    target_names=class_names_str,
                    output_dict=True
                )
                print(classification_report(y_val_encoded, y_val_pred, target_names=class_names_str))
                
                # Generate confusion matrix on validation data
                cm = confusion_matrix(y_val_encoded, y_val_pred)
                
                # Display confusion matrix in console
                self._print_confusion_matrix(cm)
                
                results['val_accuracy'] = val_accuracy
                results['classification_report'] = report
                results['confusion_matrix'] = cm
            else:
                # If no validation set, show training confusion matrix
                print("\nClassification Report (Training Data):")
                class_names_str = [str(name) for name in self.class_names]
                report = classification_report(
                    y_train, y_train_pred, 
                    target_names=class_names_str,
                    output_dict=True
                )
                print(classification_report(y_train, y_train_pred, target_names=class_names_str))
                
                cm = confusion_matrix(y_train, y_train_pred)
                self._print_confusion_matrix(cm)
                
                results['classification_report'] = report
                results['confusion_matrix'] = cm
        
        return results
    
    def _print_confusion_matrix(self, cm: np.ndarray):
        """
        Print confusion matrix in a formatted table to console.
        
        Args:
            cm: Confusion matrix
        """
        class_labels = [str(name) for name in self.class_names]
        n_classes = len(class_labels)
        
        # Calculate column widths
        max_label_width = max(len(label) for label in class_labels)
        col_width = max(6, max_label_width + 2)
        
        print("\n" + "=" * 80)
        print("CONFUSION MATRIX")
        print("=" * 80)
        print("\nRows = True Class (from data)")
        print("Columns = Predicted Class (by model)")
        print("Diagonal = Correct predictions\n")
        
        # Print header
        header = " " * (col_width + 2) + "Predicted →"
        print(header)
        
        # Print column labels
        col_labels = " " * (col_width + 2)
        for label in class_labels:
            col_labels += f"{label:>{col_width}}"
        print(col_labels)
        
        # Print separator
        separator = " " * (col_width + 2) + "-" * (col_width * n_classes)
        print(separator)
        
        # Print matrix rows
        for i, true_label in enumerate(class_labels):
            # Calculate label width (ensure it's positive)
            label_width = max(1, col_width - 7)
            value_width = max(1, col_width - 2)
            
            if i == 0:
                row_str = f"True ↓ {true_label:>{label_width}} |"
            else:
                row_str = f"       {true_label:>{label_width}} |"
            
            for j in range(n_classes):
                value = cm[i, j]
                
                # Highlight diagonal (correct predictions)
                if i == j:
                    row_str += f" [{value:{value_width}}]"
                else:
                    row_str += f"  {value:{value_width}} "
            print(row_str)
        
        # Print summary statistics
        print("\n" + "-" * 80)
        
        # Calculate per-class accuracy
        total_per_class = cm.sum(axis=1)
        correct_per_class = np.diag(cm)
        
        print("\nPer-Class Statistics:")
        print(f"{'Class':<10} {'Total':<10} {'Correct':<10} {'Accuracy':<10}")
        print("-" * 40)
        
        for i, label in enumerate(class_labels):
            total = total_per_class[i]
            correct = correct_per_class[i]
            acc = correct / total if total > 0 else 0
            print(f"{label:<10} {total:<10} {correct:<10} {acc:<10.2%}")
        
        # Overall accuracy
        total_correct = np.trace(cm)
        total_samples = cm.sum()
        overall_acc = total_correct / total_samples if total_samples > 0 else 0
        
        print("-" * 40)
        print(f"{'Overall':<10} {total_samples:<10} {total_correct:<10} {overall_acc:<10.2%}")
        print("=" * 80)
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot (optional)
        """
        # Convert class names to strings if they're numeric
        class_labels = [str(name) for name in self.class_names]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """
        Plot feature importance (for tree-based models).
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the plot (optional)
        """
        if self.model_type != 'random_forest':
            print("Feature importance only available for random_forest model")
            return
        
        # Get feature importances
        importances = self.pipeline.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 6))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_model(self, save_dir: str, model_name: str = None):
        """
        Save the trained model and metadata.
        
        Args:
            save_dir: Directory to save the model
            model_name: Custom model name (optional)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate model name with timestamp
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_type}_{timestamp}"
        
        model_path = save_dir / f"{model_name}.pkl"
        
        # Save model and metadata
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'class_names': self.class_names
        }
        
        joblib.dump(model_data, model_path)
        print(f"\nModel saved to {model_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
        """
        model_data = joblib.load(model_path)
        
        self.pipeline = model_data['pipeline']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.class_names = model_data['class_names']
        
        print(f"Model loaded from {model_path}")
        print(f"Model type: {self.model_type}")
        print(f"Classes: {self.class_names}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predicted labels, prediction probabilities)
        """
        if self.pipeline is None:
            raise ValueError("Model not trained or loaded")
        
        y_pred_encoded = self.pipeline.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            y_proba = self.pipeline.named_steps['classifier'].predict_proba(
                self.pipeline.named_steps['scaler'].transform(X)
            )
        else:
            y_proba = None
        
        return y_pred, y_proba

