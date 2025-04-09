"""
A script for training and evaluating transformer models for sequence classification,
using Ray Tune for hyperparameter optimization, with functionality for model prediction,
training loss visualization, and saving statistics.
"""
import argparse
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, Subset
from transformers import get_scheduler, AutoTokenizer
from captum.attr import IntegratedGradients


import ray
from ray import tune
from ray._private.utils import get_ray_temp_dir
from ray.air import session
from ray.train import Checkpoint

from config_ray import search_space
from interest.dataprocessor.preprocessor import TextPreprocessor
from interest.dataprocessor.dataloader import DataSetCreator
from interest.sentiment_analyser.transformer_trainer import TransformerTrainer

RANDOM_STATE = 42
K_FOLD = 5


def train_transformer(config: dict, train_dataset: torch.utils.data.Dataset,
                      model_name: str,
                      output_dir: str, num_labels: int,
                      model_path: str,
                      class_weights: torch.FloatTensor) -> None:
    """
    Train a transformer model with Ray Tune optimization, logging the training
    and validation losses.

    Args:
        config (dict): Configuration dictionary containing hyperparameters such as
        learning rate, batch size, etc.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        # val_dataset (torch.utils.data.Dataset): The validation dataset.
        model_name (str): The model name (e.g., from Huggingface) for the transformer model.
        output_dir (str): The directory where model checkpoints and loss data will be saved.
        num_labels (int): The number of labels in the classification task.
        model_path (str): The path of the mlm model.
        class_weights (torch.FloatTensor): weight of classes


    Returns:
        None
    """
    config_str = (json.dumps(config, separators=(',', ':')).replace("{", "").
                  replace("}", "").replace(":", "_").replace(
        ",", "_"))
    checkpoint_dir = Path(output_dir) / f"checkpoint_{config_str}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_STATE)
    print(f"Checkpoint Directory: {checkpoint_dir.resolve()}")
    num_epochs = config["epochs"]
    patience = config["patience"]
    hidden_dropout = config["hidden_dropout"]
    attention_dropout = config["attention_dropout"]

    train_losses = np.zeros((K_FOLD, num_epochs))
    val_losses = np.zeros((K_FOLD, num_epochs))
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f'Fold {fold + 1}/{K_FOLD}')
        trainer = TransformerTrainer(
            model_name=model_name,
            num_labels=num_labels,
            output_dir=output_dir,
            freeze=config["freeze"],
            class_weights=class_weights,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            mlm_model_path=model_path
        )

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
        eval_loader = DataLoader(val_subset, batch_size=config["batch_size"])

        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=config["lr"],
                                      weight_decay=config["weight_decay"])
        num_training_steps = config["epochs"] * len(train_loader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps
        )

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(config["epochs"]):

            train_loss = trainer.train_epoch(train_loader, optimizer, lr_scheduler)
            statistics = trainer.evaluate(eval_loader)

            train_losses[fold, epoch] = train_loss
            val_loss = round(statistics['val_loss'], 3)
            val_losses[fold, epoch] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
                torch.save(trainer.model.state_dict(), checkpoint_path)
                ray_checkpoint = Checkpoint.from_directory(str(checkpoint_dir))
                print('ray_checkpoint', ray_checkpoint)

                session.report({
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "accuracy": statistics.get('accuracy', None),
                    "f1_score": statistics.get('f1_score', None),
                    "precision_macro": statistics.get('precision_macro', None),
                    "recall_macro": statistics.get('recall_macro', None),
                    "epoch": epoch
                }, checkpoint=ray_checkpoint)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"Early stopping patience count: {epochs_without_improvement}/{patience}")

                if epochs_without_improvement >= patience:
                    print(f"Stopping early at epoch {epoch + 1} due to no improvement.")
                    break

    avg_train_losses = np.mean(train_losses, axis=0)
    std_train_losses = np.std(train_losses, axis=0)
    avg_val_losses = np.mean(val_losses, axis=0)
    std_val_losses = np.std(val_losses, axis=0)
    loss_history_path = Path(checkpoint_dir) / "loss_history.json"
    with open(loss_history_path, "w", encoding="utf-8") as f:
        json.dump({"train_losses": avg_train_losses.tolist(),
                   "std_train_losses": std_train_losses.tolist(),
                   "val_losses": avg_val_losses.tolist(),
                   "std_val_losses": std_val_losses.tolist()}, f)


def read_json_and_plot(root_dir: str) -> None:
    """
    Traverse subfolders in `root_dir`, read a JSON file containing training and validation
    losses, and save the loss plot.

    Args:
        root_dir (str): The root directory to search for JSON files with loss data.

    Returns:
        None
    """
    for sub_folder in Path(root_dir).rglob('*'):
        if sub_folder.is_dir():
            json_files = list(sub_folder.glob('*loss_history*.json'))
            if json_files:
                json_file = json_files[0]
                try:
                    with open(json_file, 'r', encoding="utf-8") as f:
                        data = json.load(f)
                        train_losses = np.array(data['train_losses'])
                        val_losses = np.array(data['val_losses'])
                        std_train_losses = np.array(data['std_train_losses'])
                        std_val_losses = np.array(data['std_val_losses'])

                        if train_losses.size > 0 and val_losses.size > 0:
                            plot_file_path = json_file.with_suffix('.png')
                            plot_losses_epochs(train_losses, val_losses, std_train_losses,
                                               std_val_losses, plot_file_path)
                            print(f"Saved loss plot for {json_file} at {plot_file_path}")
                        else:
                            print(f"No valid loss data in {json_file} at {sub_folder}")
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")


def plot_losses_epochs(train_losses: NDArray[np.float64], val_losses: NDArray[np.float64],
                       std_train_losses: NDArray[np.float64], std_val_losses: NDArray[np.float64],
                       plot_file_path: Path) -> None:
    """
    Plot the training and validation losses over epochs and save the plot to the specified file.

    Args:
        train_losses (list): A list of training losses across epochs.
        val_losses (list): A list of validation losses across epochs.
        std_train_losses (list): A list of std of training losses across epochs.
        std_val_losses (list): A list of std of validation losses across epochs.
        plot_file_path (Path): The file path where the plot will be saved.

    Returns:
        None
    """

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.fill_between(epochs, train_losses - std_train_losses, train_losses + std_train_losses,
                     alpha=0.2)

    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.fill_between(epochs, val_losses - std_val_losses, val_losses + std_val_losses, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch ")
    plt.legend()
    plt.savefig(plot_file_path)


def save_statistics(statistics: dict, output_dir: Path,
                    filename: str = "prediction_statistics.json") -> None:
    """
    Save prediction statistics (e.g., accuracy, F1-score) to a JSON file.

    Args:
        statistics (dict): The dictionary containing the prediction statistics.
        output_dir (str): The output directory where the statistics JSON file will be saved.
        filename (str): The name of the output JSON file (default is
        "prediction_statistics.json").

    Returns:
        None
    """
    serializable_stats = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in statistics.items()
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=4)

    print(f"Statistics saved to {file_path}")


def save_attribution_plot(tokens, scores, output_dir, filename):
    """
    Save a bar plot of the top 20 token attributions.

    Args:
        tokens (list): Tokens corresponding to the input IDs.
        scores (np.array): Attribution scores for each token. Can be multi-dimensional.
        output_dir (str): Directory to save the plot.
        filename (str): Filename for the plot.

    Returns:
        None
    """
    if len(scores.shape) > 1:
        scores = scores.sum(axis=1)  # Sum attribution scores across embedding dimensions

    # Get the absolute values of scores to identify the most important tokens
    abs_scores = [abs(score) for score in scores]

    # Get the top 20 tokens and their scores
    top_indices = sorted(range(len(abs_scores)), key=lambda i: abs_scores[i], reverse=True)[:20]
    top_tokens = [tokens[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    # Create a bar plot for the top 20 tokens
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(top_tokens)), top_scores, color='blue', alpha=0.6)
    plt.xticks(range(len(top_tokens)), top_tokens, rotation=45, ha='right', fontsize=8)
    plt.title("Top 20 Token-Level Attributions")
    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    plt.savefig(output_path / filename)
    plt.close()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for training and prediction tasks.

    Returns:
        argparse.Namespace: A namespace containing parsed arguments.
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--test_fp', type=str, default="",
                        help='path to test set')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='path to output')
    parser.add_argument('--model_name', type=str, required=True,
                        help='model name in huggingface')
    parser.add_argument('--lowercase', type=bool, default=False,
                        help='lowercase in preprocessing')
    parser.add_argument('--num_labels', type=int, required=True,
                        help='number of classes')
    parser.add_argument('--chunk_method', type=str, default='sliding_window',
                        help='model name in huggingface')
    parser.add_argument('--text_field_name', type=str, default='text',
                        help='field name with text')
    parser.add_argument('--label_field_name', type=str, default='label',
                        help='field name with label')

    parser.add_argument('--max_length', type=int, default=512,
                        help='max length of the input text')
    parser.add_argument('--stride', type=int, default=256,
                        help='stride of sliding window')

    main_parser = ArgumentParser(description="Main parser")
    subparsers = main_parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train", parents=[parser])
    parser_train.add_argument('--model_path', type=str, default="",
                              help='model path of a checkpoint')
    parser_train.add_argument('--train_fp', type=str, default="",
                              help='path to train set')
    parser_predict = subparsers.add_parser("predict", parents=[parser])
    parser_predict.add_argument('--freeze', type=bool, default=False,
                                help='freeze first layers while fine-tuning')
    parser_predict.add_argument('--batch_size', type=int, default=16,
                                help='batch_size')
    parser_predict.add_argument('--model_path', type=str, required=True,
                                help='model path of a checkpoint')

    parser_exp = subparsers.add_parser("explain", parents=[parser])
    parser_exp.add_argument('--freeze', type=bool, default=False,
                            help='freeze first layers while fine-tuning')
    parser_exp.add_argument('--batch_size', type=int, default=16,
                            help='batch_size')
    parser_exp.add_argument('--model_path', type=str, required=True,
                            help='model path of a checkpoint')
    parser_exp.add_argument('--use_exp', type=bool, default=False,
                            help='use explainability')
    return main_parser.parse_args()


def predict(args: argparse.Namespace) -> None:
    """
    Run prediction on a dataset using a pretrained model, and save the statistics.

    Args:
        args (argparse.Namespace): Parsed arguments from the command-line input.

    Returns:
        None
    """

    preprocessor = TextPreprocessor(model_name=args.model_name, max_length=args.max_length,
                                    lowercase=args.lowercase)
    data_loader = DataSetCreator(train_fp=Path(""), test_fp=args.test_fp)

    # _, _, test_dataset = data_loader.create_datasets(
    _, test_dataset = data_loader.create_datasets(
        label_col=args.label_field_name, text_col=args.text_field_name, method=args.chunk_method,
        window_size=args.max_length,
        stride=args.stride, preprocessor=preprocessor
    )
    if test_dataset is None:
        print("No test dataset found. Skipping prediction.")
        return

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = TransformerTrainer(
        model_name=args.model_name,
        num_labels=args.num_labels,
        output_dir=args.output_dir,
        freeze=args.freeze
    )
    trainer.load_model(args.model_path)
    trainer.model.eval()
    probabilities: list[float] = []
    labels = []

    # Disable gradient computation
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            batch = {k: v.squeeze(1).to(trainer.device).long() if k in ['input_ids',
                                                                        'attention_mask',
                                                                        'token_type_ids']
                     else v.to(trainer.device).long() for k, v in batch.items()}

            outputs = trainer.model(**batch)
            logits = outputs.logits

            prob = torch.softmax(logits, dim=-1)
            probabilities.extend(prob.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())

    statistics = trainer.make_stats(labels, probabilities)
    print(statistics)
    save_statistics(statistics, args.output_dir,
                    filename="prediction_statistics_" + Path(args.model_path).parent.name + ".json"
                    )


def explain_predict(args: argparse.Namespace) -> None:
    """
    Run prediction on a dataset using a pretrained model, and save the statistics.

    Args:
        args (argparse.Namespace): Parsed arguments from the command-line input.

    Returns:
        None
    """

    def custom_forward(input_embeddings, attention_mask):
        outputs = model(
            inputs_embeds=input_embeddings,  # Use embeddings instead of input_ids
            attention_mask=attention_mask,
        )
        return outputs.logits  # Return the logits directly

    preprocessor = TextPreprocessor(model_name=args.model_name, max_length=args.max_length,
                                    lowercase=args.lowercase)
    data_loader = DataSetCreator(train_fp=Path(""), test_fp=args.test_fp)
    _, test_dataset = data_loader.create_datasets(
        label_col=args.label_field_name, text_col=args.text_field_name, method=args.chunk_method,
        window_size=args.max_length,
        stride=args.stride, preprocessor=preprocessor
    )
    if test_dataset is None:
        print("No test dataset found. Skipping prediction.")
        return
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = TransformerTrainer(
        model_name=args.model_name,
        num_labels=args.num_labels,
        output_dir=args.output_dir,
        freeze=args.freeze
    )
    trainer.load_model(args.model_path)
    trainer.model.eval()
    probabilities: list[float] = []
    labels = []
    attributions = []

    if args.use_exp:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = trainer.model
        ig = IntegratedGradients(custom_forward)

    random.seed(42)

    # Select one random article per class
    articles_by_class = {}
    for batch in test_loader:
        for idx, label in enumerate(batch['labels']):
            label = label.item()
            if label not in articles_by_class:
                # Get all indices for the current class within the batch
                indices_for_class = [i for i, lbl in enumerate(batch['labels'])
                                     if lbl.item() == label]

                # Randomly select one index
                random_idx = random.choice(indices_for_class)

                # Select the article corresponding to the randomly selected index
                single_article = {k: v[random_idx:random_idx + 1] for k, v in batch.items()}
                articles_by_class[label] = single_article

    # Disable gradient computation
    with torch.no_grad():
        for label, batch in articles_by_class.items():
            batch = {k: v.squeeze(1).to(trainer.device).long() if k in ['input_ids',
                                                                        'attention_mask',
                                                                        'token_type_ids']
                     else v.to(trainer.device).long() for k, v in batch.items()}

            outputs = trainer.model(**batch)
            logits = outputs.logits

            prob = torch.softmax(logits, dim=-1)
            probabilities.extend(prob.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())

            if args.use_exp:
                embedding_layer = trainer.model.get_input_embeddings()
                input_embeddings = embedding_layer(batch['input_ids'])

                attributions_batch, _ = ig.attribute(
                    inputs=input_embeddings,
                    target=batch['labels'],
                    additional_forward_args=batch['attention_mask'],
                    n_steps=50,
                    return_convergence_delta=True,
                )

                for idx, attribution in enumerate(attributions_batch):
                    token_attributions = attribution.cpu().detach().numpy()
                    tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][idx].cpu().numpy())
                    attribution_details = list(zip(tokens, token_attributions))
                    attributions.append(attribution_details)

                save_attribution_plot(tokens, attributions_batch[idx].cpu().detach().numpy(),
                                      args.output_dir, f"attributions_class_{label}.png")


def train(args: argparse.Namespace) -> None:
    """
    Train a transformer model using Ray Tune with preprocessed datasets.

    Args:
        args (argparse.Namespace): Parsed arguments from the command-line input.

    Returns:
        None
    """
    ray.init()
    session_dir = get_ray_temp_dir()
    print('session_dir', session_dir)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    preprocessor = TextPreprocessor(model_name=args.model_name, max_length=args.max_length,
                                    lowercase=args.lowercase)
    data_loader = DataSetCreator(train_fp=args.train_fp, test_fp=Path(""))

    # train_dataset, val_dataset, _ = data_loader.create_datasets(
    train_dataset, _ = data_loader.create_datasets(
        label_col=args.label_field_name, text_col=args.text_field_name, method=args.chunk_method,
        window_size=args.max_length,
        stride=args.stride, preprocessor=preprocessor
    )

    class_weights = data_loader.calculate_class_weights()
    cpu_count = os.cpu_count()
    resources = {"cpu": (cpu_count - 1) if cpu_count is not None else 1}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resources["gpu"] = 1 if device == "cuda" else 0
    analysis = tune.run(
        tune.with_parameters(
            train_transformer,
            train_dataset=train_dataset,
            # val_dataset=val_dataset,
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_labels=args.num_labels,
            model_path=args.model_path,
            class_weights=class_weights
        ),
        config=search_space,
        resources_per_trial=resources,
        metric="val_loss",
        mode="min",
        raise_on_failed_trial=False
    )

    best_trial = analysis.get_best_trial(metric="val_loss", mode="min")
    print("Best trial config: ", best_trial.config)
    session_dir_latest = session_dir + '/session_latest/artifacts/'
    read_json_and_plot(session_dir_latest)


if __name__ == "__main__":
    argsparams = parse_arguments()
    if argsparams.mode == "train":
        train(argsparams)
    elif argsparams.mode == "predict":
        predict(argsparams)
    elif argsparams.mode == "explain":
        explain_predict(argsparams)
    else:
        raise Exception("Error argument!")
