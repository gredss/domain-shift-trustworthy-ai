"""
Data Manager Module for IndoBERT Clickbait Detection System

This module handles dataset loading, validation, stratified splitting,
preprocessing, and domain organization for multi-domain clickbait detection.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages dataset loading, validation, splitting, and preprocessing
    for multi-domain clickbait detection tasks.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "indobenchmark/indobert-base-p1",
        max_length: int = 128,
        random_seed: int = 42
    ):
        """
        Initialize the DataManager.
        
        Args:
            tokenizer_name: Name or path of the tokenizer to use
            max_length: Maximum sequence length for tokenization
            random_seed: Random seed for reproducibility
        """
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.random_seed = random_seed
        self.tokenizer = None
        self.raw_data = None
        self.domains = []
        
        np.random.seed(random_seed)
        logger.info(f"DataManager initialized with seed={random_seed}, max_length={max_length}")
    @classmethod
    def from_dataset_directory(
        cls,
        dataset_dir: str = "dataset",
        tokenizer_name: str = "indobenchmark/indobert-base-p1",
        max_length: int = 128,
        random_seed: int = 42
    ) -> 'DataManager':
        """
        Create a DataManager instance and load all domain CSV files from a directory.
        
        Expected files: technology.csv, politic.csv, health.csv, sport.csv, education.csv
        
        Args:
            dataset_dir: Path to directory containing domain CSV files
            tokenizer_name: Name or path of the tokenizer to use
            max_length: Maximum sequence length for tokenization
            random_seed: Random seed for reproducibility
            
        Returns:
            DataManager instance with loaded datasets
            
        Raises:
            FileNotFoundError: If dataset directory or required files not found
        """
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Define expected domain files
        domain_files = {
            'Technology': 'technology.csv',
            'Politics': 'politic.csv',
            'Health': 'health.csv',
            'Sport': 'sport.csv',
            'Education': 'education.csv'
        }
        
        csv_paths = []
        domain_names = []
        
        for domain_name, filename in domain_files.items():
            filepath = os.path.join(dataset_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Required domain file not found: {filepath}")
            csv_paths.append(filepath)
            domain_names.append(domain_name)
        
        # Create instance and load datasets
        manager = cls(tokenizer_name=tokenizer_name, max_length=max_length, random_seed=random_seed)
        manager.load_datasets(csv_paths, domain_names)
        
        return manager
    
    
    def load_datasets(self, csv_paths: List[str], domain_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load multiple CSV files (one per domain) and combine them.
        
        CSV Schema: id, source, date, title, Label, url
        The 'title' column is used as text input, 'Label' as the target.
        
        Args:
            csv_paths: List of paths to CSV files
            domain_names: Optional list of domain names corresponding to each CSV.
                         If None, extracts from filename (e.g., 'technology.csv' -> 'technology')
            
        Returns:
            Combined DataFrame with all domains
            
        Raises:
            FileNotFoundError: If any CSV file is not found
            ValueError: If CSV structure is invalid
        """
        logger.info(f"Loading {len(csv_paths)} CSV files")
        
        if domain_names and len(domain_names) != len(csv_paths):
            raise ValueError(
                f"Number of domain names ({len(domain_names)}) must match "
                f"number of CSV paths ({len(csv_paths)})"
            )
        
        dataframes = []
        for idx, path in enumerate(csv_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV file not found: {path}")
            
            df = pd.read_csv(path)
            self._validate_dataframe(df, path)
            
            # Extract domain name from filename if not provided
            if domain_names:
                domain = domain_names[idx]
            else:
                domain = os.path.splitext(os.path.basename(path))[0].capitalize()
            
            # Rename columns to standardized format
            df = df.rename(columns={'title': 'text', 'Label': 'label'})
            df['domain'] = domain
            
            dataframes.append(df)
            logger.info(f"Loaded {len(df)} samples from {os.path.basename(path)} (domain: {domain})")
        
        self.raw_data = pd.concat(dataframes, ignore_index=True)
        self.domains = sorted(self.raw_data['domain'].unique().tolist())
        
        logger.info(f"Total samples loaded: {len(self.raw_data)}")
        logger.info(f"Domains identified: {self.domains}")
        logger.info(f"Label distribution: {self.raw_data['label'].value_counts().to_dict()}")
        
        return self.raw_data
    
    def _validate_dataframe(self, df: pd.DataFrame, source: str) -> None:
        """
        Validate DataFrame structure and content.
        
        Expected CSV schema: id, source, date, title, Label, url
    
        Args:
            df: DataFrame to validate
            source: Source file path for error messages
        Raises:
            ValueError: If validation fails
        """
        required_columns = ['id', 'source', 'date', 'title', 'Label', 'url']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {source}: {missing_columns}. "
                f"Expected columns: {required_columns}"
            )
        
        if df['title'].isnull().any():
            raise ValueError(f"Found null values in 'title' column in {source}")
        
        if not df['Label'].isin([0, 1]).all():
            raise ValueError(f"Labels must be binary (0 or 1) in {source}")
    
    def stratified_split_by_domain( #NEW
        self,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Perform stratified splitting INDEPENDENTLY for each domain to support 
        specialist models and true cross-domain testing.
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_datasets() first.")
        
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError(f"Split proportions must sum to 1.0. Got: {train_size + val_size + test_size}")
        
        logger.info("Performing independent stratified splits per domain...")
        domain_splits = {}
        val_proportion = val_size / (train_size + val_size)
        
        for domain in self.domains:
            domain_df = self.raw_data[self.raw_data['domain'] == domain].copy()
            
            train_val_df, test_df = train_test_split(
                domain_df,
                test_size=test_size,
                stratify=domain_df['label'], 
                random_state=self.random_seed
            )
            
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_proportion,
                stratify=train_val_df['label'],
                random_state=self.random_seed
            )
            
            domain_splits[domain] = {
                'train': train_df,
                'val': val_df,
                'test': test_df
            }
            
            logger.info(f"  {domain} Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
        return domain_splits

    def export_domain_splits( #NEW
        self, 
        domain_splits: Dict[str, Dict[str, pd.DataFrame]],
        output_dir: str
    ) -> None:
        """
        Export domain-specific train, val, and test splits to separated CSV folders.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for domain, splits in domain_splits.items():
            domain_dir = os.path.join(output_dir, domain.lower())
            os.makedirs(domain_dir, exist_ok=True)
            
            splits['train'].to_csv(os.path.join(domain_dir, 'train.csv'), index=False)
            splits['val'].to_csv(os.path.join(domain_dir, 'val.csv'), index=False)
            splits['test'].to_csv(os.path.join(domain_dir, 'test.csv'), index=False)
            
        logger.info(f"Exported all isolated domain splits to {output_dir}")
    
    def load_domain_splits( #NEW
        self,
        splits_dir: str
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load previously exported domain-specific train, val, and test splits.
        """
        if not os.path.exists(splits_dir):
            raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
            
        logger.info(f"Loading existing domain splits from {splits_dir}")
        domain_splits = {}
        
        # Iterate through each domain folder (e.g., 'politics', 'sport')
        for domain_folder in os.listdir(splits_dir):
            domain_path = os.path.join(splits_dir, domain_folder)
            
            if os.path.isdir(domain_path):
                # Capitalize to match your original domain naming convention
                domain_name = domain_folder.capitalize() 
                domain_splits[domain_name] = {}
                
                # Load all three splits for this domain
                for split_type in ['train', 'val', 'test']:
                    file_path = os.path.join(domain_path, f"{split_type}.csv")
                    if os.path.exists(file_path):
                        domain_splits[domain_name][split_type] = pd.read_csv(file_path)
                    else:
                        logger.warning(f"Missing {split_type}.csv for domain {domain_name}")
                
                logger.info(f"  Loaded {domain_name} splits")
                        
        return domain_splits
    
    def _log_split_statistics( 
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """Log statistics about the data splits."""
        logger.info("Split Statistics:")
        logger.info(f"  Training set: {len(train_df)} samples")
        logger.info(f"  Validation set: {len(val_df)} samples")
        logger.info(f"  Test set: {len(test_df)} samples")
        
        for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            label_dist = df['label'].value_counts(normalize=True)
            logger.info(f"  {split_name} label distribution: {label_dist.to_dict()}")
            
            domain_dist = df['domain'].value_counts()
            logger.info(f"  {split_name} domain distribution: {domain_dist.to_dict()}")
    
    def initialize_tokenizer(self) -> None:
        """Initialize the tokenizer for text preprocessing."""
        logger.info(f"Initializing tokenizer: {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        logger.info("Tokenizer initialized successfully")
    
    def preprocess_texts(
        self,
        texts: List[str],
        padding: str = "max_length",
        truncation: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess texts using the tokenizer.
        
        Args:
            texts: List of text strings to preprocess
            padding: Padding strategy ('max_length', 'longest', or False)
            truncation: Whether to truncate sequences exceeding max_length
        Returns:
            Dictionary containing input_ids, attention_mask, and token_type_ids
        Raises:
            RuntimeError: If tokenizer not initialized
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call initialize_tokenizer() first.")
        
        logger.info(f"Preprocessing {len(texts)} texts")
        
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="np"
        )
        
        result = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        
        if 'token_type_ids' in encodings:
            result['token_type_ids'] = encodings['token_type_ids']
        
        logger.info(f"Preprocessing complete. Shape: {result['input_ids'].shape}")
        
        return result
    
    def organize_by_domain(
        self,
        df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Organize DataFrame by domain for domain-specific operations.
        Args:
            df: DataFrame to organize
        Returns:
            Dictionary mapping domain names to their respective DataFrames
        """
        logger.info("Organizing data by domain")
        
        domain_data = {}
        for domain in self.domains:
            domain_df = df[df['domain'] == domain].copy()
            domain_data[domain] = domain_df
            logger.info(f"  Domain '{domain}': {len(domain_df)} samples")
        
        return domain_data
    
    def get_domain_statistics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate statistics for each domain in the dataset.
        Args:
            df: DataFrame to analyze
        Returns:
            Dictionary with statistics for each domain
        """
        statistics = {}
        
        for domain in self.domains:
            domain_df = df[df['domain'] == domain]
            
            statistics[domain] = {
                'total_samples': len(domain_df),
                'clickbait_count': (domain_df['label'] == 1).sum(),
                'non_clickbait_count': (domain_df['label'] == 0).sum(),
                'clickbait_ratio': (domain_df['label'] == 1).mean(),
                'avg_text_length': domain_df['text'].str.len().mean(),
                'median_text_length': domain_df['text'].str.len().median()
            }
        
        return statistics
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the loaded dataset.
        
        Returns:
            Dictionary containing dataset summary information
        """
        if self.raw_data is None:
            return {"status": "No data loaded"}
        
        return {
            "total_samples": len(self.raw_data),
            "num_domains": len(self.domains),
            "domains": self.domains,
            "label_distribution": self.raw_data['label'].value_counts().to_dict(),
            "samples_per_domain": self.raw_data['domain'].value_counts().to_dict(),
            "max_length": self.max_length,
            "random_seed": self.random_seed,
            "tokenizer": self.tokenizer_name
        }


class DatasetValidator:
    """
    Utility class for validating dataset integrity and quality.
    """
    
    @staticmethod
    def validate_balance(df: pd.DataFrame, threshold: float = 0.3) -> bool:
        """
        Check if dataset is reasonably balanced.
        Args:
            df: DataFrame to validate
            threshold: Maximum acceptable deviation from 50-50 split
        Returns:
            True if balanced within threshold, False otherwise
        """
        label_ratio = df['label'].mean()
        deviation = abs(label_ratio - 0.5)
        
        is_balanced = deviation <= threshold
        
        if not is_balanced:
            logger.warning(
                f"Dataset imbalance detected. Label ratio: {label_ratio:.2%}, "
                f"deviation: {deviation:.2%}"
            )
        
        return is_balanced
    
    @staticmethod
    def check_text_quality(df: pd.DataFrame) -> Dict[str, int]:
        """
        Check for potential text quality issues.
        Args:
            df: DataFrame to check
        Returns:
            Dictionary with counts of various quality issues
        """
        issues = {
            'empty_texts': (df['text'].str.strip() == '').sum(),
            'very_short_texts': (df['text'].str.len() < 10).sum(),
            'very_long_texts': (df['text'].str.len() > 500).sum(),
            'duplicate_texts': df['text'].duplicated().sum()
        }
        
        for issue_type, count in issues.items():
            if count > 0:
                logger.warning(f"Found {count} instances of {issue_type}")
        
        return issues
    
    @staticmethod
    def validate_domain_distribution(df: pd.DataFrame) -> bool:
        """
        Check if domains are reasonably distributed.
        Args:
            df: DataFrame to validate
        Returns:
            True if distribution is acceptable, False otherwise
        """
        domain_counts = df['domain'].value_counts()
        min_count = domain_counts.min()
        max_count = domain_counts.max()
        
        ratio = min_count / max_count if max_count > 0 else 0
        is_balanced = ratio >= 0.5
        
        if not is_balanced:
            logger.warning(
                f"Domain distribution imbalance detected. "
                f"Min: {min_count}, Max: {max_count}, Ratio: {ratio:.2%}"
            )
        
        return is_balanced
