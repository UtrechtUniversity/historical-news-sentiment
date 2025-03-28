"""Script for loading text data from CSV files, preprocessing it,
and creating PyTorch datasets for machine learning models."""
from collections import Counter
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import numpy as np


class TextDataset(Dataset):
    """
    A PyTorch Dataset for handling text data with preprocessing
    and segmentation.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list,
        preprocessor,
        label_col: str,
        method: str,
        window_size: int,
        stride: int,
    ):
        """
        Initializes the TextDataset object.

        Args:
            texts (list): List of text documents.
            labels (list): List of corresponding labels.
            preprocessor (TextPreprocessor): Instance of the text preprocessor.
            label_col (str): Name of the column containing labels.
            method (str): Text segmentation method
                         ('sliding_window' or 'chunking').
            window_size (int): Maximum segment length in tokens.
            stride (int): Step size between windows (only for sliding window).
        """
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.label_col = label_col
        self.method = method
        self.window_size = window_size
        self.stride = stride
        self.tokenized_data = self.tokenize_texts()

    def tokenize_texts(self) -> list[tuple[dict, int]]:
        """
        Tokenizes and segments the input texts using the specified method.

        Returns:
            list[tuple[dict, int]]: A list of tokenized text segments and
            their labels.
        """
        tokenized_segments = []
        segment_labels = []

        for text, label in zip(self.texts, self.labels):
            segments = self.preprocessor.preprocess_and_split(
                text,
                method=self.method,
                window_size=self.window_size,
                stride=self.stride
            )
            for segment in segments:
                tokens = self.preprocessor.tokenize(segment)
                tokenized_segments.append(tokens)
                segment_labels.append(label)

        return list(zip(tokenized_segments, segment_labels))

    def __len__(self) -> int:
        """
        Returns:
            int: The number of tokenized segments in the dataset.
        """
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a tokenized segment and its label at the specified index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            dict: A dictionary containing tokenized inputs and the label.
        """
        tokens, label = self.tokenized_data[idx]
        tokens['labels'] = torch.tensor(label, dtype=torch.long)
        return tokens


class CSVDataLoader:
    """
    A data loader for loading and splitting CSV files into training,
    validation, and test datasets.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initializes the CSVDataLoader object.

        Args:
            csv_files (list[str]): Paths to CSV file.
            test_size (float): Proportion of the data to use for
              testing (default: 0.2).
            random_state (int): Random seed for reproducibility (default: 42).
        """
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self, data_fp) -> pd.DataFrame:
        """
        Load data from a CSV file into a pandas DataFrame.

        Args:
            data_fp (str): The file path to the CSV file.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        dataframes = pd.read_csv(data_fp)
        return dataframes

    def split_data(self,
                   data: pd.DataFrame,
                   labels: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and test sets and returns them as DataFrames.

        Args:
            data (pd.DataFrame): pandas DataFrame containing the text data.
            labels (Union[pd.Series, pd.DataFrame]): pandas Series or DataFrame of
            corresponding labels.

        Returns:
            tuple: Training and test dataframes and their respective labels as DataFrames.
        """
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=self.test_size, random_state=self.random_state
        )
        train_df = pd.DataFrame(train_data)
        train_df['label'] = train_labels
        test_df = pd.DataFrame(test_data)
        test_df['label'] = test_labels

        return train_df, test_df


class DataSetCreator:
    """
    Handles the creation of training, validation, and test datasets from CSV files.
    """
    def __init__(self, train_fp: Path, test_fp: Path):
        self.train_fp = train_fp
        self.test_fp = test_fp
        self.train_labels: Optional[np.ndarray] = None

    def create_datasets(
        self,
        label_col: str,
        text_col: str,
        method: str,
        window_size: int,
        stride: int,
        preprocessor
    ) -> tuple[Optional[TextDataset], Optional[TextDataset]]:
        """
        Creates PyTorch datasets for training, validation, and testing.

        Args:
            label_col (str): Name of the column containing labels.
            text_col (str): Name of the column containing text data.
            method (str): Text segmentation method
              ('sliding_window' or 'chunking').
            window_size (int): Maximum segment length in tokens.
            stride (int): Step size between windows (only for sliding window).
            preprocessor (TextPreprocessor): Instance of the text preprocessor.

        Returns:
            tuple[Optional[TextDataset], Optional[TextDataset]]:
              Training, and test datasets.
        """
        train_dataset = None
        test_dataset = None

        csv_dataloader = CSVDataLoader()
        if self.train_fp != "":
            data_train = csv_dataloader.load_data(self.train_fp)
            # self.train_labels = data_train[label_col].values
            self.train_labels = data_train[label_col].to_numpy()
            train_texts = data_train[text_col].values
            if self.train_labels is not None:
                train_dataset = TextDataset(
                    train_texts.astype(str).tolist(),
                    self.train_labels.astype(int).tolist(),
                    preprocessor, label_col,
                    method=method, window_size=window_size, stride=stride)
            else:
                raise ValueError(f"Column '{label_col}' in training data is None!")

        if self.test_fp != "":
            data_test = csv_dataloader.load_data(self.test_fp)
            test_labels = data_test[label_col].values
            test_texts = data_test[text_col].values
            if test_labels is not None:
                test_dataset = TextDataset(
                    test_texts.astype(str).tolist(),
                    test_labels.astype(int).tolist(), preprocessor, label_col,
                    method=method, window_size=window_size, stride=stride
                )
            else:
                raise ValueError(f"Column '{label_col}' in test data is None!")
        return train_dataset, test_dataset

    def calculate_class_weights(self):
        """
        Calculate class weights.

        Returns:
            torch.tensor: Tensor containing the class weights.
        """
        class_counts = Counter(self.train_labels)
        total_samples = len(self.train_labels)
        num_classes = len(class_counts)
        class_weights = {}

        for cls, count in class_counts.items():
            class_weights[cls] = total_samples / (num_classes * count)

        weights_tensor = torch.tensor(
            [class_weights[cls] for cls in sorted(class_weights)],
            dtype=torch.float
        )

        return weights_tensor
