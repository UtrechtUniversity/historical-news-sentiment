from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # type: ignore  # noqa:E501
from sklearn.svm import SVC  # type: ignore
from sklearn.naive_bayes import ComplementNB  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore  # noqa:E501
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Union, Any, Optional
from lime.lime_text import LimeTextExplainer  # type: ignore
import spacy  # type: ignore
import spacy.cli  # type: ignore
import json
import logging
from interest.utils.logging_utils import setup_logging
import os
import joblib
import numpy as np
from numpy.typing import NDArray


setup_logging()
logger = logging.getLogger(__name__)
logger.info("Logging initialized")


class Classifier:
    """
    A class for training and evaluating various traditional
    classifiers on text data.
    """

    def __init__(self) -> None:
        """
        Initialize the vetorizer object.
        """
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1))

        try:
            self.nlp = spacy.load('nl_core_news_sm')
        except OSError:
            logger.info("Model not found. Downloading...")
            spacy.cli.download('nl_core_news_sm')
            self.nlp = spacy.load('nl_core_news_sm')

    def _strip_classifier_prefix(
            self,
            params: Union[Dict[str, Any], List[Tuple[str, Any]]],
            valid_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Strip the 'classifier__' prefix from parameter keys
          and optionally filter to valid keys.
        """
        if isinstance(params, list):
            params = dict(params)
        stripped = {key.split("classifier__")[-1]: val for key, val in params.items()}  # noqa
        if valid_keys:
            stripped = {k: v for k, v in stripped.items() if k in valid_keys}
        return stripped

    def train_classifiers(
            self,
            text_train_vectorized: Union[List[str], List[int], List[float]],
            label_train: Union[List[str], List[int], List[float]],
            binary_labels: bool = True,
            model_dir: str = "./model",
    ) -> Dict[str, object]:
        """
        Train multiple classifiers on the training data and save them to disk.

        Parameters:
        - text_train_vectorized: Vectorized training data.
        - label_train: Training labels.
        - binary_labels: Whether it's a binary classification task.
        - model_dir: Path to directory for saving trained models.

        Returns:
        - Dict[str, object]: Dictionary of trained classifier objects.
        """

        results_file = (
            "hyperparameter_results_binary.json"
            if binary_labels
            else "hyperparameter_results_multiclass.json"
        )

        try:
            with open(results_file, 'r') as f:
                best_params = json.load(f)
        except FileNotFoundError:
            logger.info(
                f"Best parameters file '{results_file}' not found. Ensure hyperparameter "  # noqa
                "optimization is completed."
            )
            return {}

        classifiers: Dict[str, Any] = {
            "Gradient Boosting": GradientBoostingClassifier(
                **self._strip_classifier_prefix(
                    best_params.get(
                        "Gradient Boosting", {}).get("best_params", {}),
                    valid_keys=[
                        "n_estimators",
                        "learning_rate",
                        "subsample",
                        "min_samples_split",
                        "min_samples_leaf",
                        "max_depth",
                        "max_features"
                    ]
                ),
                random_state=42,
            ),
            "Support Vector Machine": SVC(
                **self._strip_classifier_prefix(
                    best_params.get(
                        "Support Vector Machine", {}).get("best_params", {}),
                    valid_keys=[
                        "C", "kernel", "degree", "gamma",
                        "coef0", "shrinking", "probability"
                    ]
                ),
                class_weight="balanced",
                random_state=42,
                probability=True,
            ),
            "Logistic Regression": LogisticRegression(
                **self._strip_classifier_prefix(
                    best_params.get(
                        "Logistic Regression", {}).get("best_params", {}),
                    valid_keys=[
                        "penalty", "C", "solver", "l1_ratio"
                    ]
                ),
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                **self._strip_classifier_prefix(
                    best_params.get(
                        "Random Forest", {}).get("best_params", {}),
                    valid_keys=[
                        "n_estimators",
                        "max_depth",
                        "min_samples_split",
                        "min_samples_leaf",
                        "max_features",
                        "bootstrap"
                    ]
                ),
                class_weight="balanced",
                random_state=42,
            ),
            "Naive Bayes": ComplementNB(
                **self._strip_classifier_prefix(
                    best_params.get("Naive Bayes", {}).get("best_params", {}),
                    valid_keys=[
                        "alpha", "norm"
                    ]
                )
            ),
        }

        # Make sure model_dir exists
        model_dir = os.path.abspath(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        trained_classifiers: Dict[str, Any] = {}
        label_type = "binary" if binary_labels else "multi"

        for clf_name, classifier in classifiers.items():
            print(f"Training {clf_name}...")
            try:
                classifier.fit(text_train_vectorized, label_train)
                trained_classifiers[clf_name] = classifier

                # Save the model
                file_safe_name = clf_name.replace(" ", "_").lower()
                model_path = os.path.join(
                    model_dir,
                    f"{file_safe_name}_{label_type}_model.joblib"
                )
                joblib.dump(classifier, model_path)
                logger.info(f"Saved {clf_name} to {model_path}")
            except Exception as e:
                logger.info(f"Error occurred while training {clf_name}: {e}")

        return trained_classifiers

    def evaluate_classifiers(
        self,
        trained_classifiers: Dict[str, Any],
        text_test_vectorized: Union[List[str], List[int], List[float]],
        label_test: List[int]
    ) -> Tuple[List[float], List[float]]:
        """
        Evaluate trained classifiers on the test data and print evaluation
          metrics.
        For binary classification, also compute FPR and TPR for ROC plotting.

        Parameters:
        - trained_classifiers (Dict[str, Any]):
          Dictionary of trained classifier objects.
        - text_test_vectorized (Union[List[str], List[int],
          List[float]]): Vectorized test set.
        - label_test (Union[List[str], List[int], List[float]]): Ground
          truth labels for test set.

        Returns:
        - Tuple[List[float], List[float]]: Lists of false positive rates
          and true positive rates
        (only populated for binary classification; empty lists
          for multi-class).
        """
        fpr_all: List[float] = []
        tpr_all: List[float] = []
        classes = sorted(set(label_test))
        is_multiclass = len(classes) > 2

        for clf_name, classifier in trained_classifiers.items():
            print(f"Evaluating {clf_name}...")
            try:
                label_predicted = classifier.predict(text_test_vectorized)
                if hasattr(classifier, "predict_proba"):
                    label_pred_proba = classifier.predict_proba(
                        text_test_vectorized
                    )
                else:
                    label_pred_proba = classifier.decision_function(
                        text_test_vectorized)

                self.print_evaluation_metrics(
                    label_test, label_predicted, label_pred_proba)

                if not is_multiclass:
                    fpr, tpr, _ = roc_curve(label_test, label_pred_proba[:, 1])
                    fpr_all.append(fpr)
                    tpr_all.append(tpr)
            except Exception as e:
                logger.info(f"Error occurred while evaluating {clf_name}: {e}")

        return fpr_all, tpr_all

    def print_evaluation_metrics(
        self,
        label_test: List[int],
        label_predicted: List[int],
        label_pred_proba: NDArray[np.float64]
    ) -> None:
        """
        Print classification report, confusion matrix, and AUC-ROC score.

        Parameters:
        - label_test (array-like): True labels.
        - label_predicted (array-like): Predicted class labels.
        - label_pred_proba (array-like): Predicted probabilities
          or decision scores.

        Returns:
        - None
        """
        try:
            print("Classification Report:")
            print(classification_report(
                label_test, label_predicted, zero_division=1))
            print(
                classification_report(
                    label_test,
                    label_predicted,
                    zero_division=1
                )
            )
            print("\nConfusion Matrix:")
            print(confusion_matrix(label_test, label_predicted))

            classes = sorted(set(label_test))
            if len(classes) > 2:
                auc_roc = roc_auc_score(
                    label_test,
                    label_pred_proba,
                    multi_class='ovr',
                    average='macro'
                )
            else:
                auc_roc = roc_auc_score(label_test, label_pred_proba[:, 1])

            print(f"AUC-ROC: {auc_roc:.4f}")
            print('\n', '***************************************', '\n')
        except Exception as e:
            logger.info(
                f"Error occurred while printing evaluation metrics: {e}")

    def plot_roc_curves(self, fpr_all: List[float], tpr_all: List[float], classifiers: Dict[str, object]) -> None:  # noqa: E501
        """
        Plot ROC curves for each classifier.

        Parameters:
        - fpr_all (list): List of false positive rates.
        - tpr_all (list): List of true positive rates.
        - classifiers (Dict): Dictionary of trained classifiers.

        Returns:
        - None
        """
        try:
            plt.figure()
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

            for i, clf_name in enumerate(classifiers.keys()):
                if i < len(fpr_all):
                    plt.plot(fpr_all[i], tpr_all[i], lw=2, label=clf_name)

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic Curve')
            plt.legend(loc="lower right")
            plt.show()
        except Exception as e:
            logger.info(f"Error occurred while plotting ROC curves: {e}")

    def train_and_evaluate_classifiers(
        self,
        text_train,
        text_test,
        label_train,
        label_test,
        binary: bool = True,
        model_dir: str = "../models"
    ) -> None:
        """
        End-to-end training and evaluation of multiple classifiers.

        Parameters:
        - text_train (List[str]): Raw text data for training.
        - text_test (List[str]): Raw text data for testing.
        - label_train (List[int] or List[str]): Training labels.
        - label_test (List[int] or List[str]): Test labels.
        - binary (bool): Whether the task is binary classification.
        - model_dir (str): Directory to save/load models.

        Returns:
        - None
        """
        try:
            text_train_vectorized = self.vectorizer.fit_transform(text_train)
            text_test_vectorized = self.vectorizer.transform(text_test)

            trained_classifiers = self.train_classifiers(
                text_train_vectorized,
                label_train,
                binary_labels=binary,
                model_dir=model_dir
            )

            fpr_all, tpr_all = self.evaluate_classifiers(
                trained_classifiers,
                text_test_vectorized,
                label_test
            )

            if binary:
                self.plot_roc_curves(fpr_all, tpr_all, trained_classifiers)

        except Exception as e:
            logger.info(f"Error occurred during training and evaluation: {e}")

    def explain_with_lime(
        self,
        text_sample: str,
        label_sample: int,
        model_dir: str = "./model",
        binary: bool = True
    ) -> None:
        """
        Use LIME to explain predictions of classifiers loaded from disk.

        Parameters:
        - text_sample (str): A single text sample to explain.
        - label_sample (int): True label for the sample.
        - model_dir (str): Directory from which to load saved models.
        - binary (bool): Whether it's a binary classification task.

        Returns:
        - None
        """
        from IPython.display import display, HTML  # ✅ safe import

        print(
            f"Actual label: {'Positive' if label_sample == 1 else 'Negative'}"
        )

        explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

        model_dir = os.path.abspath(model_dir)
        label_type = "binary" if binary else "multi"

        classifier_names = [
            "gradient_boosting",
            "support_vector_machine",
            "logistic_regression",
            "random_forest",
            "naive_bayes"
        ]

        explain_dir = os.path.abspath("explainability_outputs")
        os.makedirs(explain_dir, exist_ok=True)

        for file_safe_name in classifier_names:
            model_path = os.path.join(
                model_dir, f"{file_safe_name}_{label_type}_model.joblib")
            clf_name = file_safe_name.replace("_", " ").title()

            try:
                classifier = joblib.load(model_path)
                print(f"\nExplaining prediction for {clf_name}...\n")

                def predict_proba(texts):
                    vectorized_texts = self.vectorizer.transform(texts)
                    return classifier.predict_proba(vectorized_texts)

                explanation = explainer.explain_instance(
                    text_sample,
                    predict_proba,
                    num_features=10
                )

                custom_html = f"""
                <div style="background-color: white; color: black; padding: 10px;">  # noqa
                    {explanation.as_html()}
                </div>
                """
                display(HTML(custom_html))

                output_path = os.path.join(
                    explain_dir,
                    f"{file_safe_name}_lime_explanation.html"
                )

                explanation.save_to_file(output_path)
                logger.info(
                    f"LIME explanation for {clf_name} saved to {output_path}")

            except Exception as e:
                logger.info(
                    "Error occurred while explaining with LIME for %s: %s",
                    clf_name,
                    e
                )
