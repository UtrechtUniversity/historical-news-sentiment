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

        # self.cores = multiprocessing.cpu_count()

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


    def plot_word_vectors(self, positive_word_vector: np.ndarray, negative_word_vector: np.ndarray):
        """Reduce dimensions using PCA and plot the positive and negative word vectors."""
        
        # Combine the positive and negative word vectors for PCA
        vectors = np.vstack([positive_word_vector, negative_word_vector])

        # Use PCA to reduce dimensionality to 2 components for visualization
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)

        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_vectors[0, 0], reduced_vectors[0, 1], color='green', label='Positive Sentiment')
        plt.scatter(reduced_vectors[1, 0], reduced_vectors[1, 1], color='red', label='Negative Sentiment')

        # Label the plot
        plt.title('2D PCA of Positive and Negative Word Vectors')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

    def calculate_similarity(self, articles_word_vectors, negative_sentiment_word_vector, positive_sentiment_word_vector):
        # Check dimensionality
        vector_dim = positive_sentiment_word_vector.shape[0]
        if negative_sentiment_word_vector.shape[0] != vector_dim:
            raise ValueError("Dimensionality mismatch: positive and negative sentiment vectors must have the same dimensions.")
        
        # Initialize lists to store positive and negative similarities
        pos_similarities = []
        neg_similarities = []
        article_sentiment = []

        for article in articles_word_vectors:
            for sentence_vector in article:
                # Convert sentence_vector to a NumPy array
                sentence_vector = np.array(sentence_vector)

                # Check dimensionality
                if len(sentence_vector) != vector_dim:
                    print(len(sentence_vector))
                    print(vector_dim)
                    raise ValueError("Dimensionality mismatch: sentence vectors must have the same dimensions as the sentiment vectors.")
                
                # Reshape vectors to be 2D (necessary for cosine_similarity)
                sentence_vector_reshaped = sentence_vector.reshape(1, -1)
                positive_vector_reshaped = positive_sentiment_word_vector.reshape(1, -1)
                negative_vector_reshaped = negative_sentiment_word_vector.reshape(1, -1)
                
                # Calculate cosine similarity for each sentence with both positive and negative sentiment vectors
                pos_similarity = cosine_similarity(sentence_vector_reshaped, positive_vector_reshaped)[0][0]
                neg_similarity = cosine_similarity(sentence_vector_reshaped, negative_vector_reshaped)[0][0]

                # Append similarities to the respective lists
                pos_similarities.append(pos_similarity)
                neg_similarities.append(neg_similarity)

            # Compute the mean similarity for the entire article
            mean_pos_similarity = np.mean(pos_similarities)
            mean_neg_similarity = np.mean(neg_similarities)
            article_sentiment.append(mean_pos_similarity - mean_neg_similarity)
        print(article_sentiment)

        # # Compare the mean similarities and print the overall sentiment for the article
        # if mean_pos_similarity > mean_neg_similarity:
        #     print(f"Article is more positive (mean_pos_similarity: {mean_pos_similarity}, mean_neg_similarity: {mean_neg_similarity}).")
        # elif mean_pos_similarity < mean_neg_similarity:
        #     print(f"Article is more negative (mean_pos_similarity: {mean_pos_similarity}, mean_neg_similarity: {mean_neg_similarity}).")
        # else:
        #     print(f"Article is neutral (mean_pos_similarity: {mean_pos_similarity}, mean_neg_similarity: {mean_neg_similarity}).")

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
    analyzer.calculate_similarity(aritcles_word_vectors, negative_sentiment_word_vector, positive_sentiment_word_vector)
    # analyzer.plot_word_vectors(pos, neg)
