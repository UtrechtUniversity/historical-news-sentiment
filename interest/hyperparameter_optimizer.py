from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.naive_bayes import ComplementNB  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer   # type: ignore
from interest.utils import prepare_data   # type: ignore
import json
import logging


# flake8: noqa
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
        grid_search = GridSearchCV(clf, param_grid[clf_name], cv=5, scoring={'f1_weighted': 'f1_weighted', 'accuracy': 'accuracy'}, refit='f1_weighted', n_jobs=-1)

        grid_search.fit(X_train, y_train)

        results[clf_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

        logger.info(f"{clf_name} optimization complete.")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_}\n")

    return results


def run_optimization_pipeline(data_dir, binary_labels):
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

    train_data, _, _, train_labels, _, _ = prepare_data(data_dir, binary_labels)
    # train_data = train_data.tolist()
    # train_labels = train_labels.tolist()

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data)

    results = hyperparameter_optimization(classifiers, param_grid, X_train, train_labels)
    results_file = "hyperparameter_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"Hyperparameter optimization results saved to {results_file}")

    for clf_name, res in results.items():
        logger.info(f"{clf_name} Results: {res}")
