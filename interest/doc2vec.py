from typing import List, Tuple, Dict, Any
from collections import Counter
import pandas as pd
import numpy as np
import gensim  # type: ignore
import nltk  # type: ignore
from nltk.tokenize import sent_tokenize  # type: ignore
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # type: ignore
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)  # type: ignore

nltk.download('punkt')
# flake8: noqa: E501

class SentimentAnalyser:
    def __init__(self, negative_words_fp: str, positive_words_fp: str, articles_fp: str, model_fp: str):  # noqa: E501
        self.negative_words_fp = negative_words_fp
        self.positive_words_fp = positive_words_fp
        self.positive_words, self.negative_words = self._load_sentiment_words()
        self.articles_fp = articles_fp
        self.articles, self.sentiment_labels = self._load_articles()
        self.w2v_pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(  # noqa: E501
            model_fp, binary=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            filename='analyser.log',
            filemode='w' 
        )

    def _load_articles(self) -> Tuple[List[List[str]], List[int]]:
        """
        Load articles from the CSV file specified by `self.articles_fp`.

        This method reads the CSV file, tokenizes the article text into sentences,
        and extracts sentiment labels.

        Returns:
            Tuple[List[List[str]], List[int]]:
                - A list where each element is a list of sentences from an article.
                - A list of sentiment labels corresponding to each article.
        """
        df = pd.read_csv(self.articles_fp)
        articles = df['text'].apply(lambda x: sent_tokenize(x)).tolist()
        sentiment_labels = df['final_label'].fillna(0)
        return articles, sentiment_labels.tolist()

    def _load_sentiment_words(self) -> Tuple[List[str], List[str]]:
        """
        Load the list of positive and negative words from the specified 'self.negative_words_fp' file paths.

        This method reads two files:
        - A file containing positive words (one word per line).
        - A file containing negative words (one word per line).

        Returns:
            Tuple[List[str], List[str]]:
                - A list of positive words.
                - A list of negative words.
        """
        with open(self.positive_words_fp, 'r') as f:
            positive_words = f.read().splitlines()
        with open(self.negative_words_fp, 'r') as f:
            negative_words = f.read().splitlines()
        return positive_words, negative_words

    def text_to_word_vectors(self) -> List[List[np.ndarray]]:
        """
        Convert the articles' sentences into word vectors using a pre-trained Word2Vec model.

        This method processes each article and each sentence within it by converting words to their 
        corresponding Word2Vec vectors. For each sentence, the average vector of its words is computed. 
        If a word is not found in the Word2Vec model, it is added to the `missing_words` set.

        Returns:
            List[List[np.ndarray]]:
                A list where each element represents an article, and within each article is a list of
                NumPy arrays representing the averaged word vectors for each sentence.
        """
        articles_word_vectors = []
        missing_words = set()

        for article in tqdm(self.articles, desc="Calculating word vectors...", unit="article"):  # noqa: E501
            article_word_vectors = []
            for sentence in article:
                sentence_word_vectors = []
                for word in sentence.split():
                    word = word.lower()
                    if word in self.w2v_pretrained_model.key_to_index:
                        sentence_word_vectors.append(self.w2v_pretrained_model[word])  # noqa: E501
                    else:
                        missing_words.add(word)

                # Calculate the average vector for the sentence
                if sentence_word_vectors:
                    avg_vector = np.mean(sentence_word_vectors, axis=0)
                    article_word_vectors.append(avg_vector)
                    logging.info(f"Processed sentence with {len(sentence_word_vectors)} word vectors.")  # noqa: E501
                else:
                    avg_vector = np.zeros(self.w2v_pretrained_model.vector_size)  # noqa: E501
                    article_word_vectors.append(avg_vector)
                    logging.warning("No valid word vectors found for a sentence; appended zero vector.")  # noqa: E501

            articles_word_vectors.append(article_word_vectors)

        if missing_words:
            logging.warning(f"{len(missing_words)} words not found in the model: {', '.join(missing_words)}")  # noqa: E501

        return articles_word_vectors

    def positive_words_to_word_vectors(self) -> np.ndarray:
        """
        Convert a list of positive words into their corresponding Word2Vec vectors.

        This method processes each positive word by converting it to its Word2Vec vector. 
        If a word is not found in the Word2Vec model, it is added to the `missing_positive_words` set. 
        The function returns the average vector of all positive words, or a zero vector if none are found.

        Returns:
            np.ndarray:
                The averaged Word2Vec vector of positive words. If no valid vectors are found,
                a zero vector of the same dimensionality as the Word2Vec vectors is returned.
        """
        positive_word_vectors = []
        missing_positive_words = set()

        for word in self.positive_words:
            word = word.lower()
            if word in self.w2v_pretrained_model.key_to_index:
                positive_word_vectors.append(self.w2v_pretrained_model[word])
            else:
                missing_positive_words.add(word)

        if positive_word_vectors:
            positive_word_vector = np.mean(positive_word_vectors, axis=0)
            logging.info(f"Processed sentence with {len(positive_word_vectors)} word vectors.")  # noqa: E501
        else:
            positive_word_vector = np.zeros(self.w2v_pretrained_model.vector_size)  # noqa: E501
            logging.warning("No valid word vectors found for a sentence; appended zero vector.")  # noqa: E501

        if missing_positive_words:
            logging.warning(f"{len(missing_positive_words)} words not found in the model: {', '.join(missing_positive_words)}")  # noqa: E501

        return positive_word_vector

    def negative_words_to_word_vectors(self) -> np.ndarray:
        """
        Converts negative words to word vectors using the pre-trained word2vec model.
        
        For each negative word, this function looks up its vector representation from the
        pre-trained word2vec model. If a word is not found, it is added to the 
        `missing_negative_words` set. The function returns the mean vector of all 
        found words. If no valid vectors are found, a zero vector of the appropriate
        size is returned.

        Returns:
            np.ndarray: The averaged word vector of the negative words, or a zero vector if no word vectors were found.
        """

        negative_word_vectors = []
        missing_negative_words = set()

        for word in self.negative_words:
            word = word.lower()
            if word in self.w2v_pretrained_model.key_to_index:
                negative_word_vectors.append(self.w2v_pretrained_model[word])
            else:
                missing_negative_words.add(word)

        if negative_word_vectors:
            negative_word_vector = np.mean(negative_word_vectors, axis=0)  # noqa: E501
            logging.info(f"Processed sentence with {len(negative_word_vectors)} word vectors.")  # noqa: E501
        else:
            negative_word_vector = np.zeros(self.w2v_pretrained_model.vector_size)  # noqa: E501
            logging.warning("No valid word vectors found for a sentence; appended zero vector.")  # noqa: E501

        if missing_negative_words:
            logging.warning(f"{len(missing_negative_words)} words not found in the model: {', '.join(missing_negative_words)}")  # noqa: E501

        return negative_word_vector

    def plot_word_vectors(self, positive_word_vector: np.ndarray, negative_word_vector: np.ndarray, articles_word_vectors: List[List[np.ndarray]]) -> None :  # noqa: E501
        """
        Reduces dimensionality of word vectors using PCA and visualizes them.

        This method reduces the dimensionality of the positive, negative, and article 
        word vectors to two components using PCA (Principal Component Analysis) for visualization. 
        The reduced vectors are then plotted on a 2D scatter plot to show the relationship between 
        the sentiment vectors (positive and negative) and the text vectors.

        Args:
            positive_word_vector (np.ndarray): Vector representing the positive sentiment.
            negative_word_vector (np.ndarray): Vector representing the negative sentiment.
            articles_word_vectors (List[List[np.ndarray]]): A list of lists of word vectors 
                                                            for the articles' content.
        Returns:
            None: The function generates a plot and does not return any value.
        """

        flattened_articles_word_vectors = [vec for sublist in articles_word_vectors for vec in sublist]  # noqa: E501
        articles_word_vectors_array = np.vstack(flattened_articles_word_vectors)  # noqa: E501
        vectors = np.vstack([positive_word_vector, negative_word_vector, articles_word_vectors_array])  # noqa: E501

        # Use PCA to reduce dimensionality to 2 components for visualization
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)

        plt.figure(figsize=(8, 6))

        # Plot text word vectors
        plt.scatter(reduced_vectors[2:, 0], reduced_vectors[2:, 1], color='blue', label='Text Word Vectors', alpha=0.6)  # noqa: E501

        # Plot positive and negative sentiment vectors
        plt.scatter(reduced_vectors[0, 0], reduced_vectors[0, 1], color='green', label='Positive Sentiment')  # noqa: E501
        plt.scatter(reduced_vectors[1, 0], reduced_vectors[1, 1], color='red', label='Negative Sentiment')  # noqa: E501

        plt.title('2D PCA of Positive, Negative, and Text Word Vectors')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

    def calculate_article_sentiment(self, 
                                articles_word_vectors: List[List[np.ndarray]], 
                                negative_sentiment_word_vector: np.ndarray, 
                                positive_sentiment_word_vector: np.ndarray, 
                                neutral_threshold: float = 0.05) -> List[int]:
        """
        Calculates the sentiment (positive, negative, or neutral) of articles based on the distance to sentiment vectors.

        For each article, this function computes the Euclidean distance between the sentence 
        vectors and the positive and negative sentiment vectors. It then classifies the article's 
        overall sentiment as positive, negative, or neutral based on the average distances. 
        Articles whose average distance difference falls within the `neutral_threshold` are 
        considered neutral.

        Args:
            articles_word_vectors (List[List[np.ndarray]]): A list of lists of word vectors for each article.
            negative_sentiment_word_vector (np.ndarray): Vector representing the negative sentiment.
            positive_sentiment_word_vector (np.ndarray): Vector representing the positive sentiment.
            neutral_threshold (float, optional): The threshold for classifying an article as neutral. 
                                                Defaults to 0.05.

        Returns:
            List[int]: A list of sentiment labels for each article where 1 represents positive, 
                    0 represents neutral, and -1 represents negative.
        """
        vector_dim = positive_sentiment_word_vector.shape[0]
        if negative_sentiment_word_vector.shape[0] != vector_dim:
            raise ValueError("Dimensionality mismatch: positive and negative sentiment vectors must have the same dimensions.")  # noqa: E501

        article_sentiments_string = []
        article_sentiments = []

        for article in articles_word_vectors:
            pos_distances = []
            neg_distances = []

            for sentence_vector in article:
                sentence_vector = np.array(sentence_vector)

                if len(sentence_vector) != vector_dim:
                    raise ValueError("Dimensionality mismatch: sentence vectors must have the same dimensions as the sentiment vectors.")  # noqa: E501

                # Calculate Euclidean distance between the sentence vector and sentiment vectors  # noqa: E501
                pos_distance = np.linalg.norm(sentence_vector - positive_sentiment_word_vector)  # noqa: E501
                neg_distance = np.linalg.norm(sentence_vector - negative_sentiment_word_vector)  # noqa: E501

                pos_distances.append(pos_distance)
                neg_distances.append(neg_distance)

            # Compute the average distance for the article
            mean_pos_distance = np.mean(pos_distances)
            mean_neg_distance = np.mean(neg_distances)

            # Classify sentiment based on distances
            distance_diff = mean_pos_distance - mean_neg_distance

            if abs(distance_diff) <= neutral_threshold:
                article_sentiments_string.append("neutral")
                article_sentiments.append(0)
            elif distance_diff > neutral_threshold:
                article_sentiments_string.append("negative")
                article_sentiments.append(-1)
            else:
                article_sentiments_string.append("positive")
                article_sentiments.append(1)

        # print(article_sentiments_string)
        return article_sentiments

    def evaluate_sentiment_predictions(self, 
                                   sentiment_labels: List[int], 
                                   article_sentiments: List[int], 
                                   sentiment_type: str = 'multi') -> Dict[str, Any]:
        """
        Evaluate the performance of the sentiment prediction using various evaluation metrics.

        This function compares the true sentiment labels with the predicted sentiment labels and
        calculates metrics such as accuracy, precision, recall, F1 score, and the confusion matrix.
        It supports both binary classification (positive vs negative) and multi-class classification
        (positive, neutral, negative).

        Args:
            sentiment_labels (List[int]): List of true sentiment labels from the labeled dataset. 
                                        (1 for positive, 0 for neutral, -1 for negative).
            article_sentiments (List[int]): List of predicted sentiment labels from the sentiment model. 
                                            (1 for positive, 0 for neutral, -1 for negative).
            sentiment_type (str, optional): Specifies the type of evaluation. 'multi' for multi-class 
                                            (default), 'binary' for binary classification (positive vs negative).

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics:
                - 'accuracy': The accuracy of the sentiment predictions.
                - 'precision': The weighted precision score.
                - 'recall': The weighted recall score.
                - 'f1_score': The weighted F1 score.
                - 'confusion_matrix': The confusion matrix of the predictions.
        """


        if len(sentiment_labels) != len(article_sentiments):
            raise ValueError("The length of sentiment_labels and article_sentiments must be the same.")

        if sentiment_type == 'binary':
            # Merge neutral (0) and positive (1) into positive (1)
            modified_sentiment_labels = [1 if label in [0, 1] else -1 for label in sentiment_labels]
            modified_article_sentiments = [1 if sentiment in [0, 1] else -1 for sentiment in article_sentiments]
            target_names = ["positive", "negative"]

            label_counts = Counter(modified_sentiment_labels)
            prediction_counts = Counter(modified_article_sentiments)

            print("Counts in modified_sentiment_labels:")
            print(f"Positive (1): {label_counts[1]}")
            print(f"Neutral (0): {label_counts[0]}")
            print(f"Negative (-1): {label_counts[-1]}")

            print("Counts in modified_article_sentiments:")
            print(f"Positive (1): {prediction_counts[1]}")
            print(f"Neutral (0): {prediction_counts[0]}")
            print(f"Negative (-1): {prediction_counts[-1]}")

            conf_matrix = confusion_matrix(modified_sentiment_labels, modified_article_sentiments, labels=[1, -1])
            print("Confusion Matrix:")
            print(conf_matrix)

            report = classification_report(modified_sentiment_labels, modified_article_sentiments, target_names=target_names)
            print("Classification Report:")
            print(report)

            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_names)
            disp.plot(cmap=plt.cm.Blues)  # type: ignore[attr-defined]
            plt.title('Confusion Matrix')
            plt.show()
        else:
            # Multi-class: Keep original labels, define target names based on unique classes
            modified_sentiment_labels = sentiment_labels
            modified_article_sentiments = article_sentiments
            target_names = ["positive", "neutral", "negative"]

            label_counts = Counter(modified_sentiment_labels)
            prediction_counts = Counter(modified_article_sentiments)

            print("Counts in modified_sentiment_labels:")
            print(f"Positive (1): {label_counts[1]}")
            print(f"Neutral (0): {label_counts[0]}")
            print(f"Negative (-1): {label_counts[-1]}")

            print("Counts in modified_article_sentiments:")
            print(f"Positive (1): {prediction_counts[1]}")
            print(f"Neutral (0): {prediction_counts[0]}")
            print(f"Negative (-1): {prediction_counts[-1]}")

            conf_matrix = confusion_matrix(modified_sentiment_labels, modified_article_sentiments, labels=[1, 0, -1])
            print("Confusion Matrix:")
            print(conf_matrix)

            report = classification_report(modified_sentiment_labels, modified_article_sentiments, target_names=target_names)
            print("Classification Report:")
            print(report)

            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_names)
            disp.plot(cmap=plt.cm.Blues)  # type: ignore[attr-defined]
            plt.title('Confusion Matrix')
            plt.show()

        accuracy = accuracy_score(modified_sentiment_labels, modified_article_sentiments)
        precision = precision_score(modified_sentiment_labels, modified_article_sentiments, average='weighted')
        recall = recall_score(modified_sentiment_labels, modified_article_sentiments, average='weighted')
        f1 = f1_score(modified_sentiment_labels, modified_article_sentiments, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
