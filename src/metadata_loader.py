"""
Metadata loading module for audio classification.
Handles loading metadata from Excel files (ID, Age, Sex, Class).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


class MetadataLoader:
    """Load and process metadata from Excel files."""
    
    def __init__(self, excel_path: str, sheet_name: str = None):
        """
        Initialize the metadata loader.
        
        Args:
            excel_path: Path to the Excel file containing metadata
            sheet_name: Name of the sheet to load (default: first sheet)
        """
        self.excel_path = Path(excel_path)
        self.sheet_name = sheet_name
        self.metadata_df = None
        self.id_column = 'ID'
        self.age_column = 'Age'
        self.sex_column = 'Sex'
        self.class_column = 'Class'
        
    def load_metadata(self) -> pd.DataFrame:
        """
        Load metadata from Excel file.
        
        Returns:
            DataFrame containing metadata
        """
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
        
        # Read Excel file
        if self.sheet_name:
            self.metadata_df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
            print(f"Loaded metadata from: {self.excel_path} (Sheet: {self.sheet_name})")
        else:
            self.metadata_df = pd.read_excel(self.excel_path)
            print(f"Loaded metadata from: {self.excel_path}")
        
        print(f"Total records: {len(self.metadata_df)}")
        print(f"Columns: {self.metadata_df.columns.tolist()}")
        
        # Validate required columns
        required_columns = [self.id_column, self.age_column, self.sex_column, self.class_column]
        missing_columns = [col for col in required_columns if col not in self.metadata_df.columns]
        
        if missing_columns:
            print(f"\nWarning: Missing columns: {missing_columns}")
            print(f"Available columns: {self.metadata_df.columns.tolist()}")
        
        return self.metadata_df
    
    def get_metadata_for_file(self, file_path: str) -> Optional[Dict]:
        """
        Get metadata for a specific audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with metadata or None if not found
        """
        if self.metadata_df is None:
            self.load_metadata()
        
        # Extract ID from filename
        # Assuming filename format includes the ID
        filename = Path(file_path).stem
        
        # Try to find matching ID in metadata
        # This might need adjustment based on actual filename format
        for _, row in self.metadata_df.iterrows():
            if str(row[self.id_column]) in filename:
                return {
                    'id': row[self.id_column],
                    'age': row[self.age_column],
                    'sex': row[self.sex_column],
                    'class': row[self.class_column]
                }
        
        return None
    
    def encode_sex(self, sex_value) -> int:
        """
        Encode sex as numeric value.
        
        Args:
            sex_value: Sex value (e.g., 'M', 'F', 'Male', 'Female', 0, 1)
            
        Returns:
            Encoded sex (0 for Female, 1 for Male)
        """
        if isinstance(sex_value, (int, float)):
            return int(sex_value)
        
        sex_str = str(sex_value).upper()
        if sex_str in ['M', 'MALE', '1']:
            return 1
        elif sex_str in ['F', 'FEMALE', '0']:
            return 0
        else:
            # Default to 0 if unknown
            return 0
    
    def prepare_metadata_features(self, metadata: Dict) -> np.ndarray:
        """
        Convert metadata to feature vector.
        
        Args:
            metadata: Dictionary with metadata fields
            
        Returns:
            Feature vector [age, sex_encoded]
        """
        age = float(metadata['age'])
        sex_encoded = self.encode_sex(metadata['sex'])
        
        return np.array([age, sex_encoded])
    
    def get_class_distribution(self) -> pd.Series:
        """
        Get distribution of classes in the metadata.
        
        Returns:
            Series with class counts
        """
        if self.metadata_df is None:
            self.load_metadata()
        
        return self.metadata_df[self.class_column].value_counts().sort_index()
    
    def get_metadata_summary(self) -> Dict:
        """
        Get summary statistics of the metadata.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.metadata_df is None:
            self.load_metadata()
        
        summary = {
            'total_records': len(self.metadata_df),
            'class_distribution': self.get_class_distribution().to_dict(),
            'age_stats': {
                'mean': self.metadata_df[self.age_column].mean(),
                'std': self.metadata_df[self.age_column].std(),
                'min': self.metadata_df[self.age_column].min(),
                'max': self.metadata_df[self.age_column].max()
            },
            'sex_distribution': self.metadata_df[self.sex_column].value_counts().to_dict()
        }
        
        return summary

