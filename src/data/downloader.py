import os
import logging
import pandas as pd
from sklearn.datasets import fetch_openml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_credit_card_data(openml_id: int, output_path: str) -> pd.DataFrame:
    """
    Downloads credit card fraud dataset from OpenML if not already downloaded.
    
    Args:
        openml_id: OpenML dataset ID.
        output_path: Path where the CSV file should be saved.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if os.path.exists(output_path):
        logger.info(f"Dataset already exists at {output_path}. Skipping download.")
        return pd.read_csv(output_path)
    
    logger.info(f"Downloading dataset (OpenML ID: {openml_id}) from OpenML...")
    
    # Create target directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Fetch from OpenML
    dataset = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
    df = dataset.frame
    
    # Save to disk
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully downloaded and saved dataset to {output_path}. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Test downloading when run directly
    download_credit_card_data(1597, "data/creditcard.csv")
