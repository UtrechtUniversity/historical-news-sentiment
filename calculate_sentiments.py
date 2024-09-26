from interest.sentiment import SentimentAnalyser
from argparse import ArgumentParser
from pathlib import Path


def parse_arguments():
    parser = ArgumentParser(
        prog="csentiment_analysis.py",
        description="Calculate sentiments")
    parser.add_argument("--negative_words_dir", required=True)
    parser.add_argument("--positive_words_dir", required=True)
    parser.add_argument("--articles_dir", required=True)
    parser.add_argument("--model_dir", required=True)

    return parser.parse_args()


def compute_sentiment(negative_words_dir, positive_words_dir, articles_dir, model_dir):
    analyzer = SentimentAnalyser(negative_words_dir, positive_words_dir, articles_dir, model_dir)
    
    negative_sentiment_word_vector = analyzer.negative_words_to_word_vectors()
    positive_sentiment_word_vector = analyzer.positive_words_to_word_vectors()
    articles_word_vectors = analyzer.text_to_word_vectors()
    analyzer.plot_word_vectors(negative_sentiment_word_vector, positive_sentiment_word_vector, articles_word_vectors)
    article_sentiments = analyzer.calculate_article_sentiment(articles_word_vectors, negative_sentiment_word_vector, positive_sentiment_word_vector, neutral_threshold=0.05)
    analyzer.evaluate_sentiment_predictions(analyzer.sentiment_labels, article_sentiments)

if __name__ == '__main__':
    args = parse_arguments()
    compute_sentiment(Path(args.negative_words_dir), Path(args.positive_words_dir),
                       Path(args.articles_dir), Path(args.model_dir))
