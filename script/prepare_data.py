import argparse
import yaml
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
from interest.dataprocessor.preprocessor import TextPreprocessor
from interest.dataprocessor.dataloader import CSVDataLoader
from interest.utils.logging_utils import setup_logging


setup_logging()
logging.info("Logging initialized")

CONFIG_PATH = Path("../config/config.yaml")

def load_config(config_path: Path) -> Dict:
    """Load configuration from a YAML file."""
    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as f:
        return yaml.safe_load(f)

def get_preprocessor(config: Dict) -> TextPreprocessor:
    """Initialize and return a text preprocessor instance."""
    return TextPreprocessor(
        model_name=config["model"]["name"],
        max_length=config["model"]["max_length"],
        lowercase=config["preprocessing"]["lowercase"],
        use_stemming=config["preprocessing"]["use_stemming"],
        use_lemmatization=config["preprocessing"]["use_lemmatization"]
    )

def prepare_data(
    data_dir: Path, 
    config: Dict, 
    text_col: Optional[str] = None, 
    label_col: Optional[str] = None, 
    binary_labels: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    Load, preprocess, and split dataset.

    Args:
        data_dir (Path): Directory containing CSV files.
        config (Dict): Configuration dictionary.
        text_col (Optional[str]): Column name for text data (default: from config).
        label_col (Optional[str]): Column name for labels (default: from config).
        binary_labels (bool): Convert labels to binary format (default: True).

    Returns:
        Tuple[pd.Series, pd.Series]: Processed text data and corresponding labels.

    Raises:
        FileNotFoundError: If no CSV files are found in the given directory.
    """
    if text_col is None:
        text_col = config["data"]["text_column"]
    if label_col is None:
        label_col = config["data"]["label_column"]

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logging.error(f"No CSV files found in: {data_dir}")
        raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")

    preprocessor = get_preprocessor(config)
    loader = CSVDataLoader(preprocessor, csv_files=csv_files)
    dataset = loader.load_data()

    if binary_labels:
        dataset["binary_label"] = dataset[label_col].replace(2, 1)
        dataset.dropna(subset=["binary_label"], inplace=True)
        label_col = "binary_label"

    dataset["processed_text"] = dataset[text_col].apply(
        lambda x: preprocessor.preprocess_text(x, full_preprocessing=True)
    )

    logging.info(f"Data preprocessing completed. Processed {len(dataset)} rows.")
    
    return dataset["processed_text"], dataset[label_col]

def parse_arguments() -> Path:
    parser = argparse.ArgumentParser(description="Load and preprocess text data.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/", 
        help="Path to the data directory containing CSV files."
    )
    args = parser.parse_args()
    return Path(args.data_dir)

if __name__ == "__main__":
    data_dir = parse_arguments()
    config = load_config(CONFIG_PATH)
    processed_text, labels = prepare_data(data_dir, config)
    logging.info("Data processing completed successfully.")
