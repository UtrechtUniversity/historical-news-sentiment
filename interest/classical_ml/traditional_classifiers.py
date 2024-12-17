from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # type: ignore  # noqa:E501
from sklearn.svm import SVC  # type: ignore
from sklearn.naive_bayes import ComplementNB  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore  # noqa:E501
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Union, Any
from lime.lime_text import LimeTextExplainer  # type: ignore
import spacy


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
            print("Model not found. Downloading...")
            spacy.cli.download('nl_core_news_sm')  # type: ignore
            self.nlp = spacy.load('nl_core_news_sm')

    def train_classifiers(
        self,
        text_train_vectorized: Union[List[str], List[int], List[float]],
        label_train: Union[List[str], List[int], List[float]],
    ) -> Dict[str, object]:

        """
        Train multiple classifiers on the training data.

        Parameters:
        - text_train_vectorized (sparse matrix): Vectorized training text data.
        - label_train (array-like): Training labels.

        Returns:
        - Dict: Trained classifiers.
        """
        import json

        try:
            with open('hyperparameter_results.json', 'r') as f:
                best_params = json.load(f)
        except FileNotFoundError:
            print("Best parameters file not found. Ensure hyperparameter "
                  "optimization is completed.")

            return {}

        classifiers: Dict[str, Any] = {
            "Gradient Boosting": GradientBoostingClassifier(
                **best_params.get("Gradient Boosting", {}).get("best_params", {}),  # noqa:E501
                random_state=42,
            ),
            "Support Vector Machine": SVC(
                **best_params.get("Support Vector Machine", {}).get("best_params", {}),  # noqa:E501
                class_weight="balanced",
                kernel="rbf",
                random_state=42,
                probability=True,
            ),
            "Logistic Regression": LogisticRegression(
                **best_params.get("Logistic Regression", {}).get("best_params", {}),  # noqa:E501
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                **best_params.get("Random Forest", {}).get("best_params", {}),
                class_weight="balanced",
                random_state=42,
            ),
            "Naive Bayes": ComplementNB(
                **best_params.get("Naive Bayes", {}).get("best_params", {})
            ),
        }

        trained_classifiers: Dict[str, Any] = {}
        for clf_name, classifier in classifiers.items():
            print(f"Training {clf_name}...")
            try:
                classifier.fit(text_train_vectorized, label_train)
                trained_classifiers[clf_name] = classifier
            except Exception as e:
                print(f"Error occurred while training {clf_name}: {e}")
        return trained_classifiers

    def evaluate_classifiers(self, trained_classifiers: Dict[str, Any], text_test_vectorized: Union[List[str], List[int], List[float]], label_test: Union[List[str], List[int], List[float]]) -> Tuple[List[float], List[float]]:  # noqa: E501
        """
        Evaluate trained classifiers on the test data.

        Parameters:
        - trained_classifiers (Dict): Dictionary of trained classifiers.
        - text_test_vectorized (sparse matrix): Vectorized test text data.
        - label_test (array-like): Test labels.

        Returns:
        - Tuple: Lists of false positive rates and true positive rates.
        """
        fpr_all: List[float] = []
        tpr_all: List[float] = []
        for clf_name, classifier in trained_classifiers.items():
            print(f"Evaluating {clf_name}...")
            try:
                label_predicted = classifier.predict(text_test_vectorized)
                self.print_evaluation_metrics(label_test, label_predicted)
                fpr, tpr, _ = roc_curve(label_test, label_predicted)
                fpr_all.append(fpr)
                tpr_all.append(tpr)
            except Exception as e:
                print(f"Error occurred while evaluating {clf_name}: {e}")
        return fpr_all, tpr_all

    def print_evaluation_metrics(self, label_test: Union[List[str], List[int], List[float]], label_predicted: Union[List[str], List[int], List[float]]) -> None:  # noqa: E501
        """
        Print evaluation metrics such as classification report,
        confusion matrix, and AUC-ROC.

        Parameters:
        - label_test (array-like): True labels.
        - label_predicted (array-like): Predicted labels.

        Returns:
        - None
        """
        try:
            print("Classification Report:")
            print(classification_report(label_test, label_predicted,
                                        zero_division=1))

            print("\nConfusion Matrix:")
            print(confusion_matrix(label_test, label_predicted))

            auc_roc = roc_auc_score(label_test, label_predicted)
            print(f"AUC-ROC: {auc_roc:.4f}")
            print('\n', '******************************************', '\n')
        except Exception as e:
            print(f"Error occurred while printing evaluation metrics: {e}")

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
                plt.plot(fpr_all[i], tpr_all[i], lw=2, label=clf_name)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic Curve')
            plt.legend(loc="lower right")
            plt.show()
        except Exception as e:
            print(f"Error occurred while plotting ROC curves: {e}")

    def train_and_evaluate_classifiers(self, text_train, text_test, label_train, label_test, binary: bool = True) -> None:  # noqa: E501
        """
        Train and evaluate classifiers on the provided data.

        Parameters:
        - train_data (DataFrame): DataFrame containing text and labels.
        - multi_label (bool): Whether the dataset has multi-labels.

        Returns:
        - None
        """
        try:

            # text_train, text_test, label_train, label_test = self.split_dataset(text_set, labels, binary)  # noqa: E501

            # print('label_train counts:', label_train.value_counts())
            # print('label_test counts:', label_test.value_counts())

            text_train_vectorized = self.vectorizer.fit_transform(text_train)
            text_test_vectorized = self.vectorizer.transform(text_test)

            trained_classifiers = self.train_classifiers(text_train_vectorized, label_train)  # noqa: E501

            fpr_all, tpr_all = self.evaluate_classifiers(trained_classifiers, text_test_vectorized, label_test)  # noqa: E501

            self.plot_roc_curves(fpr_all, tpr_all, trained_classifiers)
        except Exception as e:
            print(f"Error occurred during training and evaluation: {e}")

    def explain_with_lime(self, trained_classifiers: Dict[str, object],
                          text_sample: str, label_sample: int) -> None:
        """
        Use LIME to explain the predictions of each classifier.

        Parameters:
        - trained_classifiers (Dict): Dictionary of trained classifiers.
        - text_sample (str): A single text sample to explain.
        - label_sample (int): True label for the sample.

        Returns:
        - None
        """

        print(f"Actual label: {'Positive' if label_sample == 1 else 'Negative'}")  # noqa:E501

        explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
        for clf_name, classifier in trained_classifiers.items():
            print(f"\nExplaining prediction for {clf_name}...\n")
            try:
                def predict_proba(texts):
                    vectorized_texts = self.vectorizer.transform(texts)
                    return classifier.predict_proba(vectorized_texts)

                explanation = explainer.explain_instance(
                    text_sample,
                    predict_proba,
                    num_features=10
                )
                explanation.show_in_notebook()
                explanation.save_to_file(f"{clf_name}_lime_explanation.html")
            except Exception as e:
                print(f"Error occurred while explaining with LIME for {clf_name}: {e}")  # noqa:E501
