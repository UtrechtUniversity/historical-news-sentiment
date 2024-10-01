from typing import List, Tuple
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

    def plot_word_vectors(self, positive_word_vector: np.ndarray, negative_word_vector: np.ndarray, articles_word_vectors: List[List[np.ndarray]]):  # noqa: E501
        """Reduce dimensions using PCA and plot the positive, negative, and text word vectors."""  # noqa: E501

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

    def calculate_article_sentiment(self, articles_word_vectors, negative_sentiment_word_vector, positive_sentiment_word_vector, neutral_threshold=0.05):  # noqa: E501
        """Calculate the sentiment (positive, negative, or neutral) of articles based on the distance to sentiment vectors."""  # noqa: E501

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

    def evaluate_sentiment_predictions(self, sentiment_labels, article_sentiments, sentiment_type='multi'):
        """
        Evaluate the performance of the sentiment prediction using evaluation metrics.

        Parameters:
        - sentiment_labels: List of true sentiment labels (from the labeled dataset).
        - article_sentiments: List of predicted sentiment labels (from the sentiment model).
        - sentiment_type: 'multi' for multi-class sentiment evaluation, 'binary' for binary (positive vs negative). 

        Returns:
        - A dictionary containing accuracy, precision, recall, F1 score for each sentiment class, and the confusion matrix.
        """

        if len(sentiment_labels) != len(article_sentiments):
            raise ValueError("The length of sentiment_labels and article_sentiments must be the same.")

        if sentiment_type == 'binary':
            # Merge neutral (0) and positive (1) into positive (1)
            modified_sentiment_labels = [1 if label in [0, 1] else -1 for label in sentiment_labels]
            modified_article_sentiments = [1 if sentiment in [0, 1] else -1 for sentiment in article_sentiments]
            target_names = ["positive", "negative"]

            # ****
            label_counts = Counter(modified_sentiment_labels)
            prediction_counts = Counter(modified_article_sentiments)

            # Print counts for 1, -1, and 0
            print("Counts in modified_sentiment_labels:")
            print(f"Positive (1): {label_counts[1]}")
            print(f"Neutral (0): {label_counts[0]}")
            print(f"Negative (-1): {label_counts[-1]}")

            print("Counts in modified_article_sentiments:")
            print(f"Positive (1): {prediction_counts[1]}")
            print(f"Neutral (0): {prediction_counts[0]}")
            print(f"Negative (-1): {prediction_counts[-1]}")


            # ***

            conf_matrix = confusion_matrix(modified_sentiment_labels, modified_article_sentiments, labels=[1, -1])
            print("Confusion Matrix:")
            print(conf_matrix)

            report = classification_report(modified_sentiment_labels, modified_article_sentiments, target_names=target_names)
            print("Classification Report:")
            print(report)

            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_names)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.show()
        else:
            # Multi-class: Keep original labels, define target names based on unique classes
            modified_sentiment_labels = sentiment_labels
            modified_article_sentiments = article_sentiments
            target_names = ["positive", "neutral", "negative"]

            # ****
            label_counts = Counter(modified_sentiment_labels)
            prediction_counts = Counter(modified_article_sentiments)

            # Print counts for 1, -1, and 0
            print("Counts in modified_sentiment_labels:")
            print(f"Positive (1): {label_counts[1]}")
            print(f"Neutral (0): {label_counts[0]}")
            print(f"Negative (-1): {label_counts[-1]}")

            print("Counts in modified_article_sentiments:")
            print(f"Positive (1): {prediction_counts[1]}")
            print(f"Neutral (0): {prediction_counts[0]}")
            print(f"Negative (-1): {prediction_counts[-1]}")


            # ***

            conf_matrix = confusion_matrix(modified_sentiment_labels, modified_article_sentiments, labels=[1, 0, -1])
            print("Confusion Matrix:")
            print(conf_matrix)

            report = classification_report(modified_sentiment_labels, modified_article_sentiments, target_names=target_names)
            print("Classification Report:")
            print(report)

            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_names)
            disp.plot(cmap=plt.cm.Blues)
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
