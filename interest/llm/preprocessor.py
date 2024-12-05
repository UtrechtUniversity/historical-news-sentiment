"""Script for preprocessing, tokenizing, and segmenting text data using various methods
like sliding window and chunking."""
import re
from transformers import AutoTokenizer


class TextPreprocessor:
    """
    A class to preprocess and tokenize text data, with support for text cleaning, segmentation,
    and tokenization.
    """

    def __init__(self, model_name: str, max_length: int, lowercase: bool,
                 remove_non_ascii: bool = True):
        """
        Initializes the TextPreprocessor.

        Args:
            model_name (str): Name of the pre-trained model for tokenization.
            max_length (int): Maximum sequence length for tokenization.
            lowercase (bool): Flag to indicate if text should be converted to lowercase.
            remove_non_ascii (bool, optional): Flag to remove non-ASCII characters from text.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.remove_non_ascii = remove_non_ascii
        self.lowercase = lowercase

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the text by removing non-ASCII characters and converting to lowercase.

        Args:
            text (str): The raw text to be processed.

        Returns:
            str: The preprocessed text.
        """
        if not isinstance(text, str):
            text = ""
        if self.remove_non_ascii:
            text = re.sub(r'[^\x00-\x7F]+', '', text)
        if self.lowercase:
            text = text.lower()
        return ' '.join(text.split())

    def tokenize(self, text: str) -> dict:
        """
        Tokenizes the preprocessed text into tokens and returns them as tensors.

        Args:
            text (str): The preprocessed text to be tokenized.

        Returns:
            dict: A dictionary containing tokenized text as PyTorch tensors.
        """
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def apply_sliding_window(self, text: str, window_size: int, stride: int) -> list[str]:
        """
        Segments the text into overlapping windows of words with a specified stride.

        Args:
            text (str): The preprocessed text to be split into windows.
            window_size (int): The size of each window (number of words).
            stride (int): The step size for the sliding window.

        Returns:
            list[str]: List of text windows (substrings of the original text).
        """
        words = text.split()
        windows = []
        for i in range(0, len(words), stride):
            window = words[i:i + window_size]
            if window:
                windows.append(" ".join(window))
        return windows

    def apply_chunking(self, text: str, chunk_size: int) -> list[str]:
        """
        Splits the text into non-overlapping chunks.

        Args:
            text (str): The preprocessed text to be chunked.
            chunk_size (int): The size of each chunk (number of words).

        Returns:
            list[str]: List of text chunks (substrings of the original text).
        """
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def preprocess_and_split(self, text: str, method: str, window_size: int, stride: int) -> (
            list)[str]:
        """
        Preprocesses the text and splits it based on the specified method
        (sliding window or chunking).

        Args:
            text (str): The raw text to be preprocessed and split.
            method (str): The segmentation method ('sliding_window' or 'chunking').
            window_size (int): The size of the sliding window or chunk (in words).
            stride (int): The stride for sliding window segmentation (ignored for chunking).

        Returns:
            list[str]: A list of text segments based on the chosen segmentation method.

        Raises:
            ValueError: If an unknown method is provided.
        """
        preprocessed_text = self.preprocess_text(text)

        if method == 'sliding_window':
            return self.apply_sliding_window(preprocessed_text, window_size, stride)
        if method == 'chunking':
            return self.apply_chunking(preprocessed_text, window_size)

        raise ValueError("Unknown method: choose either 'sliding_window' or 'chunking'")
