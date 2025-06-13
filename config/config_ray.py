"""
This script defines a hyperparameter search space for training a machine learning model
using Ray Tune, a library for distributed hyperparameter tuning and experiment execution.
"""
from ray import tune

search_space = {
    "epochs": tune.choice([2]),
    "batch_size": tune.choice([16]),
    "lr": tune.loguniform(1e-5, 5e-4),
    "freeze": tune.choice([False]),
    "weight_decay": tune.choice([0.001]),
    "patience": tune.choice([2]),
    "hidden_dropout": tune.choice([0.3, 0.4]),
    "attention_dropout": tune.choice([0.3, 0.4])
}
