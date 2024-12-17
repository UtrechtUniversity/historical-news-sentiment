import logging
from pathlib import Path
from typing import Tuple, List
from interest.llm.dataloader import CSVDataLoader
from interest.llm.preprocessor import TextPreprocessor

# Constants
MODEL_NAME = "emanjavacas/GysBERT-v2"
MAX_LENGTH = 128
DEFAULT_TEXT_COL = "text"
DEFAULT_LABEL_COL = "final_label"


# Preprocessor
def get_preprocessor(
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
    lowercase: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = False
) -> TextPreprocessor:
    return TextPreprocessor(
        model_name=model_name,
        max_length=max_length,
        lowercase=lowercase,
        use_stemming=use_stemming,
        use_lemmatization=use_lemmatization
    )


PREPROCESSOR = get_preprocessor()


# Logging
def setup_logging(
    log_file: str,
    level: int = logging.INFO,
    format: str = "%(asctime)s - %(levelname)s - %(message)s",
):
    """
    Set up logging configuration.

    Args:
        log_file (str): Path to the log file.
        level (int, optional): Logging level. Defaults to logging.INFO.
        format (str, optional): Log message format. Defaults to a
        standard timestamped format.
    """
    logging.basicConfig(filename=log_file, level=level, format=format)


# Helpers
def prepare_data(
    data_dir: Path,
    binary_labels: bool = True
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Load and preprocess the dataset.

    Args:
        data_dir (Path): Directory containing CSV files.
        binary_labels (bool, optional): If True, converts
        labels to binary. If False, retains original multi-labels.

    Returns:
        Tuple: Processed train data, validation data, test data,
        and their respective labels.
    """
    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")

    loader = CSVDataLoader(PREPROCESSOR, csv_files=csv_files)
    dataset = loader.load_data()

    if binary_labels:
        dataset['binary_label'] = dataset[DEFAULT_LABEL_COL].replace(2, 1)
        dataset.dropna(subset=['binary_label'], inplace=True)
        label_column = 'binary_label'
    else:
        label_column = DEFAULT_LABEL_COL

    dataset['processed_text'] = dataset[DEFAULT_TEXT_COL].apply(
        lambda x: PREPROCESSOR.preprocess_text(x, full_preprocessing=True)
    )

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = loader.split_data(  # noqa:E501
        data=dataset['processed_text'], labels=dataset[label_column]
    )

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels  # noqa:E501


# TODO:
# - Move all logging configurations to a dedicated logging utility.
# - Add unit tests for `prepare_data`.
# - Ensure `TextPreprocessor` and `CSVDataLoader`
#  adhere to required input-output standards.
