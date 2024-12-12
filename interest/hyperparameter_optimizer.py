from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pandas as pd
from interest.llm.dataloader import CSVDataLoader
from interest.llm.preprocessor import TextPreprocessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def hyperparameter_optimization(classifiers, param_grid, X_train, y_train):
    """
    Perform hyperparameter optimization for a set of classifiers using GridSearchCV.

    Args:
        classifiers (dict): A dictionary with classifier names as keys and classifier objects as values.
        param_grid (dict): A dictionary with classifier names as keys and their parameter grids as values.
        X_train (sparse matrix): Training data features.
        y_train (array-like): Training data labels.

    Returns:
        dict: Results of the optimization containing best parameters and scores for each classifier.
    """
    results = {}
    for clf_name, clf in classifiers.items():
        logger.info(f"Starting hyperparameter optimization for {clf_name}...")
        # grid_search = GridSearchCV(clf, param_grid[clf_name], cv=5, scoring='f1_weighted', n_jobs=8)
        grid_search = GridSearchCV(clf, param_grid[clf_name], cv=5, scoring={'f1_weighted', 'accuracy'}, refit='f1_weighted', n_jobs=-1)

        grid_search.fit(X_train, y_train)

        results[clf_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

        logger.info(f"{clf_name} optimization complete.")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_}\n")

    return results

def prepare_data(preprocessor, data_dir, label_col='final_label', text_col='text'):
    """
    Load and preprocess the dataset.

    Args:
        preprocessor (TextPreprocessor): Text preprocessing object.
        data_dir (Path): Directory containing CSV files.
        label_col (str): Column name for labels in the dataset.
        text_col (str): Column name for text in the dataset.

    Returns:
        tuple: Processed train data, validation data, test data, and their respective labels.
    """
    csv_files = list(data_dir.glob('*.csv'))
    loader = CSVDataLoader(preprocessor, csv_files=csv_files)
    df_news = loader.load_data()

    # Ensure binary labels and handle missing values
    df_news['binary_label'] = df_news[label_col].replace(2, 1)
    df_news.dropna(subset=['binary_label'], inplace=True)

    # Preprocess text data
    df_news['processed_text'] = df_news[text_col].apply(
        lambda x: preprocessor.preprocess_text(x, full_preprocessing=True))

    return loader.split_data(data=df_news['processed_text'], labels=df_news['binary_label'])

def run_optimization_pipeline(data_dir, model_name, label_col='final_label', text_col='text'):
    """Execute the hyperparameter optimization pipeline."""
    # Define classifiers and their parameter grids
    classifiers = {
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(class_weight='balanced', kernel='rbf', random_state=42),
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "Naive Bayes": ComplementNB()
    }

    param_grid = {
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.5]
        },
        "Support Vector Machine": {
            "C": [0.1, 1, 10],
            "gamma": [0.1, 1, 10]
        },
        "Logistic Regression": {
            "C": [0.1, 1, 10],
            "penalty": ['l2']
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20]
        },
        "Naive Bayes": {
            "alpha": [0.1, 0.5, 1.0],
            "norm": [False, True]
        }
    }

    # Load and preprocess data
    preprocessor = TextPreprocessor(
        model_name=model_name, max_length=128, lowercase=True,
        use_stemming=False, use_lemmatization=False
    )

    train_data, _, _, train_labels, _, _ = prepare_data(preprocessor, data_dir, label_col, text_col)

    # Vectorize text data
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data)

    # Perform hyperparameter optimization
    results = hyperparameter_optimization(classifiers, param_grid, X_train, train_labels)

    # Log results
    for clf_name, res in results.items():
        logger.info(f"{clf_name} Results: {res}")

if __name__ == '__main__':
    data_dir = Path("../data/merged")
    model_name = "emanjavacas/GysBERT-v2"
    run_optimization_pipeline(data_dir=data_dir, model_name=model_name)
