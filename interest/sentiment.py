from typing import List
import  pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import logging
import multiprocessing
from time import time
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
nltk.download('punkt')

class SentimentAnalyser:
    def __init__(self, negative_words_fp: List[str], positive_words_fp: List[str], articles_fp: str):
        self.negative_words_fp = negative_words_fp
        self.positive_words_fp = positive_words_fp
        self.positive_words, self.negative_words = self._load_sentiment_words()
        self.articles_fp = articles_fp
        self.articles = self._load_articles()
        self.w2v_pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(
            "../models/pretrained/Dutch_CoNLL17_corpus/model.bin", binary=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt= '%H:%M:%S')


    def _load_articles(self) -> List[List[str]]:
        """Load articles from the CSV file and return a list of sentences."""
        df = pd.read_csv(self.articles_fp)
        articles = df['text'].apply(lambda x: sent_tokenize(x)).tolist()
        return articles  

    def _load_sentiment_words(self) -> (List[str], List[str]):
        """Load the list of positive and negative words."""
        with open(self.positive_words_fp, 'r') as f:
            positive_words = f.read().splitlines()
        with open(self.negative_words_fp, 'r') as f:
            negative_words = f.read().splitlines()
        return positive_words, negative_words

    def text_to_word_vectors(self) -> List[List[np.ndarray]]:
        articles_word_vectors = []
        missing_words = set()

        for article in tqdm(self.articles, desc="Calculating word vectors...", unit="article"):
            article_word_vectors = []
            for sentence in article:
                sentence_word_vectors = []
                for word in sentence.split():
                    word = word.lower()
                    if word in self.w2v_pretrained_model.key_to_index:
                        sentence_word_vectors.append(self.w2v_pretrained_model[word])
                    else:
                        missing_words.add(word)
                
                # Calculate the average vector for the sentence
                if sentence_word_vectors:
                    avg_vector = np.mean(sentence_word_vectors, axis=0)
                    article_word_vectors.append(avg_vector)
                    logging.info(f"Processed sentence with {len(sentence_word_vectors)} word vectors.")
                else:
                    avg_vector = np.zeros(self.w2v_pretrained_model.vector_size)
                    article_word_vectors.append(avg_vector)
                    logging.warning("No valid word vectors found for a sentence; appended zero vector.")
                    
            articles_word_vectors.append(article_word_vectors)

        if missing_words:
            logging.warning(f"{len(missing_words)} words not found in the model: {', '.join(missing_words)}")
        
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
                    logging.info(f"Processed sentence with {len(positive_word_vectors)} word vectors.")
        else:
            positive_word_vector = np.zeros(self.w2v_pretrained_model.vector_size)
            logging.warning("No valid word vectors found for a sentence; appended zero vector.")
            

        if missing_positive_words:
            logging.warning(f"{len(missing_positive_words)} words not found in the model: {', '.join(missing_positive_words)}")
        
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
                    negative_word_vector = np.mean(negative_word_vectors, axis=0)
                    logging.info(f"Processed sentence with {len(negative_word_vectors)} word vectors.")
        else:
            negative_word_vector = np.zeros(self.w2v_pretrained_model.vector_size)
            logging.warning("No valid word vectors found for a sentence; appended zero vector.")
            

        if missing_negative_words:
            logging.warning(f"{len(missing_negative_words)} words not found in the model: {', '.join(missing_negative_words)}")
        
        return negative_word_vector

    def plot_word_vectors(self, positive_word_vector: np.ndarray, negative_word_vector: np.ndarray, articles_word_vectors: List[List[np.ndarray]]):
        """Reduce dimensions using PCA and plot the positive, negative, and text word vectors."""
        
        # Flatten the list of lists of text word vectors into a single list of np.ndarrays
        flattened_articles_word_vectors = [vec for sublist in articles_word_vectors for vec in sublist]

        # Convert the flattened list of vectors to a 2D NumPy array
        articles_word_vectors_array = np.vstack(flattened_articles_word_vectors)

        # Combine positive, negative, and text word vectors for PCA
        vectors = np.vstack([positive_word_vector, negative_word_vector, articles_word_vectors_array])

        # Use PCA to reduce dimensionality to 2 components for visualization
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)

        # Create a scatter plot
        plt.figure(figsize=(8, 6))

        # Plot text word vectors
        plt.scatter(reduced_vectors[2:, 0], reduced_vectors[2:, 1], color='blue', label='Text Word Vectors', alpha=0.6)

        # Plot positive and negative sentiment vectors
        plt.scatter(reduced_vectors[0, 0], reduced_vectors[0, 1], color='green', label='Positive Sentiment')
        plt.scatter(reduced_vectors[1, 0], reduced_vectors[1, 1], color='red', label='Negative Sentiment')

        # Label the plot
        plt.title('2D PCA of Positive, Negative, and Text Word Vectors')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()

        # Show the plot
        plt.show()

    def calculate_article_sentiment(self, articles_word_vectors, negative_sentiment_word_vector, positive_sentiment_word_vector, neutral_threshold=0.05):
        """Calculate the sentiment (positive, negative, or neutral) of articles based on the distance to sentiment vectors."""
        
        vector_dim = positive_sentiment_word_vector.shape[0]
        if negative_sentiment_word_vector.shape[0] != vector_dim:
            raise ValueError("Dimensionality mismatch: positive and negative sentiment vectors must have the same dimensions.")
        
        article_sentiments = []
        
        for article in articles_word_vectors:
            pos_distances = []
            neg_distances = []
            
            for sentence_vector in article:
                # Convert sentence_vector to a NumPy array
                sentence_vector = np.array(sentence_vector)

                # Check dimensionality
                if len(sentence_vector) != vector_dim:
                    raise ValueError("Dimensionality mismatch: sentence vectors must have the same dimensions as the sentiment vectors.")
                
                # Calculate Euclidean distance between the sentence vector and sentiment vectors
                pos_distance = np.linalg.norm(sentence_vector - positive_sentiment_word_vector)
                neg_distance = np.linalg.norm(sentence_vector - negative_sentiment_word_vector)

                # Append distances to the respective lists
                pos_distances.append(pos_distance)
                neg_distances.append(neg_distance)

            # Compute the average distance for the article
            mean_pos_distance = np.mean(pos_distances)
            mean_neg_distance = np.mean(neg_distances)

            # Classify sentiment based on distances
            distance_diff = mean_pos_distance - mean_neg_distance

            if abs(distance_diff) <= neutral_threshold:
                article_sentiments.append("neutral")
            elif distance_diff > neutral_threshold:
                article_sentiments.append("negative")
            else:
                article_sentiments.append("positive")

        print(article_sentiments)
        return article_sentiments


     
if __name__ == '__main__':
    analyzer = SentimentAnalyser('../data/negative_words_gpt.txt', '../data/positive_words_gpt.txt', '../data/merged/coal/1960s_coal.csv')
    aritcles_word_vectors = analyzer.text_to_word_vectors()
    num_articles = len(aritcles_word_vectors)
    num_sentence_vectors_per_article = len(aritcles_word_vectors[0])
    sentence_vector_shape = len(aritcles_word_vectors[0][0])

    print(f"Number of articles: {num_articles}")
    print(f"Number of sentence vectors per article: {num_sentence_vectors_per_article}")
    print(f"Shape of sentence vectors: {sentence_vector_shape}")

    negative_sentiment_word_vector = analyzer.negative_words_to_word_vectors()
    positive_sentiment_word_vector = analyzer.positive_words_to_word_vectors()
    articles_word_vectors = analyzer.text_to_word_vectors()
    # analyzer.plot_word_vectors(negative_sentiment_word_vector, positive_sentiment_word_vector, articles_word_vectors)
    analyzer.calculate_article_sentiment(articles_word_vectors, negative_sentiment_word_vector, positive_sentiment_word_vector, neutral_threshold=0.05)
