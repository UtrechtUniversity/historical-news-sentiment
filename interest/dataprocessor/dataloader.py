"""Script for loading text data from CSV files, preprocessing it,
and creating PyTorch datasets for machine learning models."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore
from typing import Union


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
              data to use for validation
             (default: 0.1).
            random_state (int): Random seed for reproducibility (default: 42).
        """
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self, data_fp) -> pd.DataFrame:
        """
     
        """
        dataframes = pd.read_csv(data_fp)
        return dataframes

    def split_data(self,
                   data: Union[list, pd.Series],
                   labels: Union[list, pd.Series]) -> tuple:
        """
        Splits the data into training, validation, and test sets.

        Args:
            data (Union[list, pd.Series]): List or pandas Series
              of text data.
            labels (Union[list, pd.Series]): List or pandas Series
              of corresponding labels.

        Returns:
            tuple: Training, and test datasets and their
              respective labels.
        """
        if isinstance(data, pd.Series):
            data = data.tolist()
        if isinstance(labels, pd.Series):
            labels = labels.tolist()

        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=self.test_size,
            random_state=self.random_state
        )
    
        return (train_data, test_data,
                train_labels, test_labels)


class DataSetCreator:
    """
    """
    def __init__(self, train_fp:Path, test_fp:Path):
        self.train_fp = train_fp
        self.test_fp = test_fp

    def create_datasets(
        self,
        label_col: str,
        text_col: str,
        method: str,
        window_size: int,
        stride: int,
        preprocessor
    ) -> tuple[TextDataset, TextDataset, TextDataset]:
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
            tuple[TextDataset, TextDataset, TextDataset]:
              Training, validation, and test datasets.
        """

        csvdataloader = CSVDataLoader()

        data_train = csvdataloader.load_data(self.train_fp)
        data_test =csvdataloader.load_data(self.test_fp)
        train_labels = data_train[label_col].values
        train_texts = data_train[text_col].values
        test_labels = data_test[label_col].values
        test_texts = data_test[text_col].values

        train_texts, val_texts, train_labels, val_labels= (   # noqa: E501
            csvdataloader.split_data(
                train_texts.tolist() if hasattr(train_texts, 'tolist') else list(train_texts),
                train_labels.tolist() if hasattr(train_labels, 'tolist') else list(train_labels)))   # noqa: E501

        train_dataset = TextDataset(
            train_texts, train_labels, preprocessor, label_col,
            method=method, window_size=window_size, stride=stride
        )
        val_dataset = TextDataset(
            val_texts, val_labels, preprocessor, label_col,
            method=method, window_size=window_size, stride=stride
        )
        test_dataset = TextDataset(
            test_texts, test_labels, preprocessor, label_col,
            method=method, window_size=window_size, stride=stride
        )

        return train_dataset, val_dataset, test_dataset
