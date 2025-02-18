import argparse
import yaml
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
from interest.dataprocessor.preprocessor import TextPreprocessor
from interest.dataprocessor.dataloader import CSVDataLoader
from interest.utils.logging_utils import setup_logging
import os


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
    input_fp: Path,
    output_dir: Path,
    config: Dict, 
    text_col: Optional[str] = None, 
    label_col: Optional[str] = None, 
    binary_labels: bool = True
) -> None:
    """
    Load and split dataset.

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

    loader = CSVDataLoader()
    dataset = loader.load_data(data_fp=input_fp)
    logging.info(f"Data is loaded. Shape of dataset: {dataset.shape} .")

    if binary_labels:
        dataset["binary_label"] = dataset[label_col].replace(0, 1)
        dataset.dropna(subset=["binary_label"], inplace=True)
        label_col = "binary_label"

    preprocessor = get_preprocessor(config)
    dataset["processed_text"] = dataset[text_col].apply(
        lambda x: preprocessor.preprocess_text(x, full_preprocessing=True)
    )
    logging.info(f"Data preprocessing completed. Processed {len(dataset)} rows.")

    train_dataset, test_dataset = loader.split_data(data=dataset.drop(columns=[label_col]), labels=dataset[label_col])
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_csv(os.path.join(output_dir, 'train_dataset.csv'), index=False)
    test_dataset.to_csv(os.path.join(output_dir, 'test_dataset.csv'), index=False)
    logging.info(f"Data spliting completed. Shape of trainset: {train_dataset.shape} and the shape of testset is: {test_dataset.shape}")

    
def parse_arguments() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(description="Load and preprocess text data.")
    parser.add_argument(
        "--data_fp",  
        type=str, 
        default="data/", 
        help="Path to the input data file path."
    )
    parser.add_argument(
        "--output_dir",  
        type=str, 
        default="output/", 
        help="Path to the output directory."
    )
    args = parser.parse_args()
    return Path(args.data_fp), Path(args.output_dir)

if __name__ == "__main__":
    data_fp, output_dir = parse_arguments()
    config = load_config(CONFIG_PATH)
    prepare_data(data_fp, output_dir, config)

    logging.info("Data processing completed successfully.")