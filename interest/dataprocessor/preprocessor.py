"""Script for preprocessing, tokenizing,
and segmenting text data using various methods
like sliding window and chunking."""
import re
from transformers import AutoTokenizer  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import SnowballStemmer  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore
import nltk  # type: ignore
from interest.utils.logging_utils import setup_logging
import logging


setup_logging()
logging.info("Logging initialized")

nltk.download('stopwords')


class TextPreprocessor:
    """
    A class to preprocess and tokenize text data, with
      support for text cleaning, segmentation, and
      tokenization.
    """

    def __init__(self, model_name: str, max_length: int, lowercase: bool,
                 remove_non_ascii: bool = True, use_stemming: bool = False,
                 use_lemmatization: bool = False):
        """
        Initializes the TextPreprocessor.

        Args:
            model_name (str): Name of the pre-trained model for tokenization.
            max_length (int): Maximum sequence length for tokenization.
            lowercase (bool): Flag to indicate if text should be converted to
             lowercase.
            remove_non_ascii (bool, optional): Flag to remove non-ASCII
              characters from text.
            use_stemming (bool, optional): Flag to apply stemming on text
              for traditional models.
            use_lemmatization (bool, optional): Flag to apply lemmatization o
        """
        logging.info("Initializing TextPreprocessor with model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.remove_non_ascii = remove_non_ascii
        self.lowercase = lowercase
        self.stopwords: set[str] = set()
        self.stemmer = SnowballStemmer("dutch") if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        logging.info("TextPreprocessor initialized successfully")

    def preprocess_text(self, text: str,
                        full_preprocessing: bool = False) -> str:
        """
        Preprocesses the text by removing non-ASCII characters,
        converting to lowercase, and handling negations within
        a window.

        Args:
            text (str): The raw text to be processed.
            full_preprocessing (bool): If True, applies
              full preprocessing suitable for traditional models.
            If False, applies mild preprocessing for BERT.

        Returns:
            str: The preprocessed text.
        """
        logging.info("\n Preprocessing text: %s", text[:50])
        if not full_preprocessing:
            logging.info("Applying mild preprocessing for llms")
            if not isinstance(text, str):
                logging.warning("Input text is not a string. Setting to empty string.")
                text = ""

            if self.remove_non_ascii:
                text = re.sub(r'[^\x00-\x7F]+', '', text)
                logging.info("Removed non-ASCII characters")

            if self.lowercase:
                text = text.lower()
                logging.info("Converted text to lowercase")

            logging.info("Text preprocessing completed")
            return ' '.join(text.split())

        if full_preprocessing:
            logging.info("Applying full preprocessing for traditional models")
            if not isinstance(text, str):
                text = ""

            negations = ['niet', 'geen', 'nooit', 'niets', 'noch',
                         'niemand', 'nochthans', 'ondertussen', 'zonder']
            words = text.split()
            processed_words = []

            i = 0
            while i < len(words):
                word = words[i]
                if word in negations:
                    # Add NEG_ prefix to the negation word itself
                    processed_words.append(f"NEG_{word}")

                    # Also add NEG_ prefix to the next 2 words (window)
                    for j in range(i + 1, min(i + 2, len(words))):
                        processed_words.append(f"NEG_{words[j]}")
                    i += 3
                else:
                    processed_words.append(word)
                    i += 1

            text = ' '.join(processed_words)
            logging.info("Detected negations")

            if self.remove_non_ascii:
                text = re.sub(r'[^\x00-\x7F]+', '', text)
                logging.info("Removed non-ASCII characters")

            if self.lowercase:
                text = text.lower()
                logging.debug("Converted text to lowercase")

            text = re.sub(r'[^\w\s]', '', text)
            logging.info("Removed punctuation")
            text = ' '.join(text.split())

            if not self.stopwords:
                self.stopwords = set(stopwords.words('dutch'))
            text = ' '.join(word for word in text.split() if word
                            not in self.stopwords or word in negations)
            logging.info("Removed Dutch stopwords")

            if self.stemmer:
                text = ' '.join(self.stemmer.stem(word) for word in text.split())   # noqa: E501
                logging.info("Applied stemming")
            elif self.lemmatizer:
                text = ' '.join(self.lemmatizer.lemmatize(word) for word in text.split())   # noqa: E501
                logging.info("Applied lemmatization")
        
        logging.info("Text preprocessing completed")
        return text

    def tokenize(self, text: str) -> dict:
        """
        Tokenizes the preprocessed text into tokens and returns them
        as tensors.

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

    def apply_sliding_window(self,
                             text: str,
                             window_size: int,
                             stride: int) -> list[str]:
        """
        Segments the text into overlapping windows of words
        with a specified stride.

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
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]   # noqa: E501

    def preprocess_and_split(self, text: str, method: str, window_size: int, stride: int) -> (   # noqa: E501
            list)[str]:
        """
        Preprocesses the text and splits it based on the specified method
        (sliding window or chunking).

        Args:
            text (str): The raw text to be preprocessed and split.
            method (str): The segmentation method ('sliding_window'
            or 'chunking').
            window_size (int): The size of the sliding window or
            chunk (in words).
            stride (int): The stride for sliding window segmentation
            (ignored for chunking).

        Returns:
            list[str]: A list of text segments based on the chosen
            segmentation method.

        Raises:
            ValueError: If an unknown method is provided.
        """
        preprocessed_text = self.preprocess_text(text)

        if method == 'sliding_window':
            return self.apply_sliding_window(preprocessed_text, window_size, stride)   # noqa: E501
        if method == 'chunking':
            return self.apply_chunking(preprocessed_text, window_size)

        raise ValueError("Unknown method: choose either 'sliding_window' or 'chunking'")   # noqa: E501
