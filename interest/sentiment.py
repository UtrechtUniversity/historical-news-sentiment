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

    def text_to_word_vectors(self) -> List[np.ndarray]:
        word_vectors = []
        missing_words = []
        
        for article in self.articles:
            for sentence in article:
                for word in sentence.split():
                    if word in self.w2v_pretrained_model .key_to_index:
                        word_vectors.append(self.w2v_pretrained_model [word])
                    else:
                        missing_words.append(word)
            
        if missing_words:
            logging.warning(f"{len(missing_words)} words not found in the model: {', '.join(missing_words)}")
        
        return word_vectors

    
    # def train_word2vec(self, vector_size: int = 100, window: int = 5, min_count: int = 1):
    #     """Train a Word2Vec model on the sentences of the news articles."""
    #     # Flatten the list of lists into a single list of sentences
    #     sentences = [sentence.split() for article in self.articles for sentence in article]
    #     self.model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count)

    # def _average_vector(self, words: List[str]) -> np.ndarray:
    #     """Compute the average vector for a list of words."""
    #     vectors = [self.model.wv[word] for word in words if word in self.model.wv]
    #     if vectors:
    #         return np.mean(vectors, axis=0)
    #     else:
    #         return np.zeros(self.model.vector_size)


if __name__ == '__main__':
    analyzer = SentimentAnalyser('../data/negative_words_gpt.txt', '../data/positive_words_gpt.txt', '../data/merged/coal/1960s_coal.csv')
    print(analyzer.text_to_word_vectors()[0])
