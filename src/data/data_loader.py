import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_and_split_data(
    raw_path: str,
    train_path: str,
    test_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    target_col: str = "Class"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the raw data into train and test sets in a stratified manner,
    saving the output files to disk.
    
    Args:
        raw_path: Path to the raw CSV file.
        train_path: Path to save the training CSV file.
        test_path: Path to save the testing CSV file.
        test_size: Ratio of the test split.
        random_state: Random state for reproducibility.
        target_col: Name of the label column.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
        
    logger.info(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Check class distribution
    class_counts = df[target_col].value_counts()
    class_ratios = df[target_col].value_counts(normalize=True)
    logger.info(f"Class distribution: {dict(class_counts)} (Ratios: {dict(class_ratios)})")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Splitting data with test_size={test_size} and stratify=y...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Recombine to save
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved stratified train set ({train_df.shape}) to {train_path}")
    logger.info(f"Saved stratified test set ({test_df.shape}) to {test_path}")
    
    return train_df, test_df
