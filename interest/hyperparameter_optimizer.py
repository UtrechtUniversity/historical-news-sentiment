from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from imblearn.pipeline import Pipeline  # type: ignore
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.naive_bayes import ComplementNB  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter  # type: ignore
import json
import logging
from sklearn.metrics import make_scorer, roc_auc_score

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to check if a classifier supports predict_proba()
def supports_predict_proba(clf):
    return hasattr(clf, "predict_proba")

def hyperparameter_optimization(classifiers, param_grid, X_train, y_train, binary_labels):
    results = {}
    
    for clf_name, clf in classifiers.items():
        scoring = {'f1_weighted': 'f1_weighted' if not binary_labels else 'f1'}
        refit_metric = 'f1_weighted'
        
        if supports_predict_proba(clf):
            scoring['roc_auc'] = make_scorer(roc_auc_score, response_method='predict') if binary_labels else 'roc_auc_ovr'
            refit_metric = 'roc_auc' 
        
        class_counts = Counter(y_train)
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
        logger.info(f"Imbalance ratio for {clf_name}: {imbalance_ratio}")
        
        use_smote = (binary_labels and imbalance_ratio > 1.5) or (not binary_labels and any(v < 10 for v in class_counts.values()))
        
        pipeline_steps = [('classifier', clf)]
        if use_smote:
            pipeline_steps.insert(0, ('smote', SMOTE()))
        
        pipeline = Pipeline(pipeline_steps)

        logger.info(f"Starting hyperparameter optimization for {clf_name}...")
        param_combinations = len(list(ParameterSampler(param_grid[clf_name], min(100, len(list(ParameterSampler(param_grid[clf_name], 1)))))))
        
        # Use only the relevant scoring metrics for each classifier
        valid_scoring = {k: v for k, v in scoring.items() if k != 'roc_auc' or supports_predict_proba(clf)}
        refit_metric = 'roc_auc' if 'roc_auc' in valid_scoring else 'f1_weighted'
        
        grid_search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_grid[clf_name], 
            n_iter=min(10, param_combinations),  
            cv=3, 
            scoring=valid_scoring, 
            refit=refit_metric, 
            n_jobs=4
        )

        grid_search.fit(X_train, y_train)
        train_score = grid_search.best_estimator_.score(X_train, y_train)
        val_score = grid_search.best_score_
        logger.info(f"Training Score: {train_score}")
        logger.info(f"Cross-validation Score: {val_score}")
        
        if train_score < 0.6 and val_score < 0.6:
            logger.warning(f"{clf_name} may be underfitting: Training Score: {train_score}, Validation Score: {val_score}")
        if train_score > 0.6 and train_score > val_score + max(0.1, 0.1 * val_score):
            logger.warning(f"{clf_name} may be overfitting: Training Score: {train_score}, Validation Score: {val_score}")

        results[clf_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

        logger.info(f"{clf_name} optimization complete.")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_}\n")

    return results

def run_optimization_pipeline(train_data, train_labels, binary_labels):
    classifiers = {
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', decision_function_shape='ovr', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Naive Bayes": ComplementNB()
    }

    param_grid = {
        "Gradient Boosting": {
            "classifier__max_iter": [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.05, 0.1] if not binary_labels else [0.01, 0.1, 0.5]
        },
        "Support Vector Machine": {
            "classifier__C": [0.01, 0.1, 1] if not binary_labels else [0.1, 1, 10],
            "classifier__gamma": [0.01, 0.1, 1] if not binary_labels else [0.1, 1, 10]
        },
        "Logistic Regression": {
            "classifier__C": [0.001, 0.01, 0.1, 1] if not binary_labels else [0.01, 0.1, 1, 10],
            "classifier__penalty": ['l2']
        },
        "Random Forest": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [5, 10] if not binary_labels else [2, 5, 10]
        },
        "Naive Bayes": {
            "classifier__alpha": [0.1, 0.5, 1.0],
            "classifier__norm": [False, True]
        }
    }

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    
    if "Gradient Boosting" in classifiers:
        logger.info(f"Converting X_train to dense array for Gradient Boosting. Shape: {X_train.shape}")
        X_train = X_train.toarray()
    
    results = hyperparameter_optimization(classifiers, param_grid, X_train, train_labels, binary_labels)
    results_file = "hyperparameter_results_binary.json" if binary_labels else "hyperparameter_results_multiclass.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Hyperparameter optimization results saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to write results to {results_file}: {e}")

    for clf_name, res in results.items():
        logger.info(f"{clf_name} Results: {res}")
