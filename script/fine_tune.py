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

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

import ray
from ray import tune
from ray._private.utils import get_ray_temp_dir
from ray.air import session
from ray.train import Checkpoint

from interest.llm.preprocessor import TextPreprocessor
from interest.llm.dataloader import CSVDataLoader
from interest.llm.transformer_trainer import TransformerTrainer


def train_transformer(config: dict, train_dataset: torch.utils.data.Dataset,
                      val_dataset: torch.utils.data.Dataset, model_name: str,
                      output_dir: str, num_labels: int) -> None:
    """
    Train a transformer model with Ray Tune optimization, logging the training
    and validation losses.

    Args:
        config (dict): Configuration dictionary containing hyperparameters such as
        learning rate, batch size, etc.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        val_dataset (torch.utils.data.Dataset): The validation dataset.
        model_name (str): The model name (e.g., from Huggingface) for the transformer model.
        output_dir (str): The directory where model checkpoints and loss data will be saved.
        num_labels (int): The number of labels in the classification task.

    Returns:
        None
    """
    config_str = (json.dumps(config, separators=(',', ':')).replace("{", "").
                  replace("}", "").replace(":", "_").replace(
        ",", "_"))
    checkpoint_dir = Path(output_dir) / f"checkpoint_{config_str}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint Directory: {checkpoint_dir.resolve()}")
    trainer = TransformerTrainer(
        model_name=model_name,
        num_labels=num_labels,
        output_dir=output_dir,
        freeze=config["freeze"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=config["lr"])
    num_training_steps = config["epochs"] * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        # Training step for each epoch
        train_loss = trainer.train_epoch(train_loader, optimizer, lr_scheduler)
        statistics = trainer.evaluate(eval_loader)

        train_losses.append(train_loss)
        val_losses.append(statistics['val_loss'])

        val_loss = statistics['val_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save(trainer.model.state_dict(), checkpoint_path)
            ray_checkpoint = Checkpoint.from_directory(str(checkpoint_dir))
            print('ray_checkpoint', ray_checkpoint)

            session.report({
                "val_loss": statistics['val_loss'],
                "train_loss": train_loss,
                "accuracy": statistics.get('accuracy', None),
                "f1_score": statistics.get('f1_score', None),
                "precision_macro": statistics.get('precision_macro', None),
                "recall_macro": statistics.get('recall_macro', None),
                "epoch": epoch
            }, checkpoint=ray_checkpoint)

    loss_history_path = Path(checkpoint_dir) / "loss_history.json"
    with open(loss_history_path, "w", encoding="utf-8") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)


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
                        train_losses = data.get('train_losses', [])
                        val_losses = data.get('val_losses', [])

                        if train_losses and val_losses:
                            plot_file_path = json_file.with_suffix('.png')
                            plot_losses_epochs(train_losses, val_losses, plot_file_path)
                            print(f"Saved loss plot for {json_file} at {plot_file_path}")
                        else:
                            print(f"No valid loss data in {json_file} at {sub_folder}")
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")


def plot_losses_epochs(train_losses: list, val_losses: list, plot_file_path: Path) -> None:
    """
    Plot the training and validation losses over epochs and save the plot to the specified file.

    Args:
        train_losses (list): A list of training losses across epochs.
        val_losses (list): A list of validation losses across epochs.
        plot_file_path (Path): The file path where the plot will be saved.

    Returns:
        None
    """

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for training and prediction tasks.

    Returns:
        argparse.Namespace: A namespace containing parsed arguments.
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='path to csv files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='path to output')
    parser.add_argument('--model_name', type=str, required=True,
                        help='model name in huggingface')
    parser.add_argument('--lowercase', type=bool, default=False,
                        help='lowercase in preprocessing')
    parser.add_argument('--num_labels', type=int, default=3,
                        help='number of classes')
    parser.add_argument('--chunk_method', type=str, default='sliding_window',
                        help='model name in huggingface')
    parser.add_argument('--text_field_name', type=str, default='text',
                        help='field name with text')
    parser.add_argument('--label_field_name', type=str, default='final_label',
                        help='field name with label')

    parser.add_argument('--max_length', type=int, default=512,
                        help='max length of the input text')
    parser.add_argument('--stride', type=int, default=256,
                        help='stride of sliding window')

    main_parser = ArgumentParser(description="Main parser")
    subparsers = main_parser.add_subparsers(dest="mode")

    #  parser_train = subparsers.add_parser("train", parents=[parser])
    parser_predict = subparsers.add_parser("predict", parents=[parser])
    parser_predict.add_argument('--freeze', type=bool, default=False,
                                help='freeze first layers while fine-tuning')
    parser_predict.add_argument('--batch_size', type=int, default=16,
                                help='batch_size')
    parser_predict.add_argument('--model_path', type=str, required=True,
                                help='model path of a checkpoint')

    return main_parser.parse_args()


def predict(args: argparse.Namespace) -> None:
    """
    Run prediction on a dataset using a pretrained model, and save the statistics.

    Args:
        args (argparse.Namespace): Parsed arguments from the command-line input.

    Returns:
        None
    """
    csv_files = list(Path(args.data_dir).glob('*.csv'))

    preprocessor = TextPreprocessor(model_name=args.model_name, max_length=args.max_length,
                                    lowercase=args.lowercase)
    data_loader = CSVDataLoader(preprocessor, csv_files=csv_files)

    _, _, test_dataset = data_loader.create_datasets(
        label_col=args.label_field_name, text_col=args.text_field_name, method=args.chunk_method,
        window_size=args.max_length,
        stride=args.stride
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = TransformerTrainer(
        model_name=args.model_name,
        num_labels=args.num_labels,
        output_dir=args.output_dir,
        freeze=args.freeze
    )
    trainer.load_model(args.model_path)
    probabilities = []
    labels = []

    # Disable gradient computation
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.squeeze(1).to(trainer.device) if k in ['input_ids', 'attention_mask',
                                                                 'token_type_ids']
                     else v.to(trainer.device) for k, v in batch.items()}

            outputs = trainer.model(**batch)
            logits = outputs.logits

            prob = torch.softmax(logits, dim=-1)
            probabilities.extend(prob.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())

    statistics = trainer.make_stats(labels, probabilities)
    print(statistics)
    save_statistics(statistics, args.output_dir, filename="prediction_statistics.json")


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

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    csv_files = list(Path(args.data_dir).glob('*.csv'))

    preprocessor = TextPreprocessor(model_name=args.model_name, max_length=args.max_length,
                                    lowercase=args.lowercase)
    data_loader = CSVDataLoader(preprocessor, csv_files=csv_files)

    train_dataset, val_dataset, _ = data_loader.create_datasets(
        label_col=args.label_field_name, text_col=args.text_field_name, method=args.chunk_method,
        window_size=args.max_length,
        stride=args.stride
    )

    resources = {"cpu": 1}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resources["gpu"] = 1 if device == "cuda" else 0

    search_space = {
        "epochs": tune.choice([2]),  # 3, 5, 7
        "batch_size": tune.choice([16]),  # 16
        "lr": tune.loguniform(1e-5, 5e-4),  #
        "freeze": tune.choice([True])  # , False
    }

    # Run Ray Tune with preprocessed datasets
    analysis = tune.run(
        tune.with_parameters(
            train_transformer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_labels=args.num_labels
        ),
        config=search_space,
        resources_per_trial=resources,
        metric="val_loss",
        mode="min"
    )

    best_trial = analysis.get_best_trial(metric="val_loss", mode="min")
    print("Best trial config: ", best_trial.config)
    session_dir_latest = session_dir+'/session_latest/artifacts/'
    read_json_and_plot(session_dir_latest)


if __name__ == "__main__":
    argsparams = parse_arguments()
    if argsparams.mode == "train":
        train(argsparams)
    elif argsparams.mode == "predict":
        predict(argsparams)
    else:
        raise Exception("Error argument!")
