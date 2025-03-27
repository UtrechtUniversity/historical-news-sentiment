"""
A script to define and train transformer models for sequence classification,
including training, evaluation, and metric computation.
"""

import os
from typing import List
from sklearn import metrics  # type: ignore
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoConfig  # type: ignore

from transformers import AutoModelForMaskedLM  # type: ignore


class TransformerTrainer:
    """
        A class for training and evaluating transformer-based
        models for sequence classification tasks.
    """
    def __init__(self, model_name, num_labels, output_dir, freeze, class_weights=torch.tensor([]),
                 hidden_dropout=0.3, attention_dropout=0.3, mlm_model_path=""):

        """
        Initializes the TransformerTrainer instance,
        loads the model, and freezes layers if specified.

        Args:
            model_name (str): The name of the pre-trained model.
            num_labels (int): The number of output labels for classification.
            output_dir (str): The directory to save the model outputs.
            freeze (bool): Whether to freeze the model's layers.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout,  # Dropout for fully connected layers
            attention_probs_dropout_prob=attention_dropout  # Dropout for attention probabilities
        )

        if mlm_model_path == "":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                # num_labels=num_labels,
                config=config,
                ignore_mismatched_sizes=True
            )
        else:
            mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)

            mlm_model.load_state_dict(torch.load(mlm_model_path, map_location="cpu",
                                                 weights_only=False), strict=False)

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                # num_labels=num_labels,
                config=config,
                ignore_mismatched_sizes=True
            )
            self.model.base_model.load_state_dict(
                mlm_model.base_model.state_dict(), strict=False
            )

        if freeze:
            for param in self.model.base_model.embeddings.parameters():
                param.requires_grad = False
            for layer in self.model.base_model.encoder.layer[:6]:

                for param in layer.parameters():
                    param.requires_grad = False

        self.model.to(self.device)

        
        class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def train_epoch(self, train_loader, optimizer, lr_scheduler) -> float:
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader):
              The data loader for the training dataset.
            optimizer (Optimizer):
              The optimizer to use for updating model parameters.
            lr_scheduler (LambdaLR):
              The learning rate scheduler.

        Returns:
            float: The average loss for the epoch.
        """
        self.model.train()
        epoch_train_loss = 0.0
        for _, batch in enumerate(train_loader):
            batch = {
                k: v.squeeze(1).to(self.device)
                if k in ['input_ids', 'attention_mask', 'token_type_ids']
                else v.to(self.device)
                for k, v in batch.items()
            }

            outputs = self.model(**batch)
            labels = batch['labels']
            loss = self.criterion(outputs.logits, labels)
            #  loss = outputs.loss
            epoch_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = epoch_train_loss / len(train_loader)
        return avg_train_loss

    def evaluate(self, eval_loader: torch.utils.data.DataLoader) -> dict:
        """
        Evaluate the model on a given evaluation dataset
        and compute evaluation metrics.

        This method sets the model to evaluation mode, processes
        the dataset in batches, calculates the average loss, and
        computes metrics such as accuracy, F1-score, precision,
        and recall using the `make_stats` method.

        Args:
            eval_loader (torch.utils.data.DataLoader):
              DataLoader providing the evaluation dataset in batches.

        Returns:
            dict: A dictionary containing the evaluation metrics including:
                - `val_loss` (float):
                  The average loss across the evaluation dataset.
                - `accuracy` (float, optional):
                  The accuracy of the predictions.
                - `f1_score` (float, optional):
                  The F1 score of the predictions.
                - `precision_macro` (float, optional):
                  Macro-averaged precision.
                - `recall_macro` (float, optional):
                  Macro-averaged recall.
        """
        self.model.eval()
        total_loss = 0.0
        all_labels: List[int] = []
        all_probabilities: List[List[float]] = []

        for batch in eval_loader:
            batch = {
                k: v.squeeze(1).to(self.device)
                if k in ['input_ids', 'attention_mask', 'token_type_ids']
                else v.to(self.device)
                for k, v in batch.items()
            }

            with torch.no_grad():
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            all_labels.extend(batch['labels'].cpu().numpy().tolist())

        avg_loss = total_loss / len(eval_loader)
        labels_array = np.array(all_labels)
        probabilities_array = np.array(all_probabilities)
        statistics = self.make_stats(
            labels_array, probabilities_array, avg_loss
        )

        return statistics

    def make_stats(
            self,
            labels,
            probabilities,
            avg_loss=0.0) -> dict:
        """
        Computes classification statistics (accuracy, AUC,
        precision, recall, F1-score).

        Args:
            labels (np.ndarray): The true labels.
            probabilities (np.ndarray): The predicted probabilities.
            avg_loss (float, optional): The average loss during evaluation.

        Returns:
            dict: A dictionary containing computed statistics.
        """
        predictions = np.argmax(probabilities, axis=1)
        accuracy = np.mean(predictions == labels)
        
        if probabilities.shape[1] == 2:
            auc = metrics.roc_auc_score(labels, probabilities[:, 1])
        else:
            auc = metrics.roc_auc_score(labels, probabilities,
                                        average='macro', multi_class='ovo')

        precision_macro = metrics.precision_score(
            labels, predictions, average='macro', zero_division=0
        )
        recall_macro = metrics.recall_score(
            labels, predictions, average='macro'
        )
        f1_macro = metrics.f1_score(labels, predictions, average='macro')

        precision_score = metrics.precision_score(
            labels, predictions, average=None, zero_division=0
        )
        recall_score = metrics.recall_score(labels, predictions, average=None)
        f1_score = metrics.f1_score(labels, predictions, average=None)

        statistics = {
            'val_loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'f1_score': f1_score,
        }
        return statistics

    def load_model(self, model_path) -> None:
        """
        Loads a model from a checkpoint.

        Args:
            model_path (str): The path to the model checkpoint.
        """
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
