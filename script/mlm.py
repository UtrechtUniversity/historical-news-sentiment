"""
This module provides functionality to train a Transformer model on Masked Language Modeling (MLM)
 using a streaming dataset. It includes a custom PyTorch IterableDataset for efficient data
 loading from large CSV files and utilizes Hugging Face's Trainer API for model training.
"""
import os
import argparse
from argparse import ArgumentParser
from pathlib import Path
import multiprocessing
from transformers import (AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling,
                          TrainingArguments, Trainer)

import pandas as pd
import torch
from torch.utils.data import IterableDataset
from interest.dataprocessor.preprocessor import TextPreprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_cpu_cores = max(multiprocessing.cpu_count() - 1, 1)


# pylint: disable=abstract-method
class StreamingCSVTextDataset(IterableDataset):
    """
    A memory-efficient PyTorch dataset that streams and tokenizes text dynamically.
    """

    def __init__(self, csv_files: list[Path], text_col: str,
                 preprocessor=None, method: str = "chunking", window_size: int = 512,
                 stride: int = 256, batch_size: int = 16):
        """
        Args:
            csv_files (list[Path]): List of CSV file paths.
            text_col (str): Column name for text data.
            preprocessor (TextPreprocessor): Text processor and tokenizer.
            method (str): "chunking" for classification, "sliding_window" for MLM.
            window_size (int): Segment size (MLM).
            stride (int): Step size (MLM).
            batch_size (int): Number of rows to read at a time.
        """
        self.csv_files = csv_files
        self.text_col = text_col
        self.preprocessor = preprocessor
        self.method = method
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size

    def process_row(self, row):
        """Processes a single row: text cleaning, segmentation, tokenization."""
        text = row[self.text_col]
        segments = self.preprocessor.preprocess_and_split(text, self.method,
                                                          self.window_size, self.stride)

        for segment in segments:
            tokenized = self.preprocessor.tokenize(segment)
            tokenized['input_ids'] = tokenized['input_ids'].squeeze(0)  # From [1, 512] -> [512]
            tokenized['attention_mask'] = tokenized['attention_mask'].squeeze(0)
            tokenized['token_type_ids'] = tokenized['token_type_ids'].squeeze(0)
            yield {k: v.to(device) for k, v in tokenized.items()}

    def stream_csv_data(self):
        """Streams CSV data in chunks to avoid memory overload."""
        for file in self.csv_files:
            df_iter = pd.read_csv(file, usecols=[self.text_col],
                                  chunksize=self.batch_size, iterator=True)
            for chunk in df_iter:
                for _, row in chunk.iterrows():
                    yield from self.process_row(row)

    def __iter__(self):
        return self.stream_csv_data()


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
    parser.add_argument('--lowercase', action='store_true',
                        help='Convert text to lowercase')

    parser.add_argument('--chunk_method', type=str, default='sliding_window',
                        help='model name in huggingface')
    parser.add_argument('--text_field_name', type=str, default='text',
                        help='field name with text')

    parser.add_argument('--max_length', type=int, default=512,
                        help='max length of the input text')

    parser.add_argument('--stride', type=int, default=256,
                        help='stride of sliding window')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')

    return parser.parse_args()


def train_mlm(args):
    """Trains a Transformer model on Masked Language Modeling using a streaming dataset."""
    preprocessor = TextPreprocessor(args.model_name, max_length=args.max_length,
                                    lowercase=args.lowercase)

    csv_files = list(Path(args.data_dir).glob("*.csv"))

    dataset = StreamingCSVTextDataset(csv_files=csv_files, text_col=args.text_field_name,
                                      preprocessor=preprocessor, method=args.chunk_method,
                                      window_size=args.max_length, stride=args.stride,
                                      batch_size=args.batch_size)

    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=preprocessor.tokenizer,
        mlm=True,
        mlm_probability=0.15,
        return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        max_steps=100,
        save_steps=10,
        save_total_limit=2,
        num_train_epochs=1,
        report_to="tensorboard",
        logging_steps=500,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # Mixed precision training for faster performance
        dataloader_num_workers=num_cpu_cores if num_cpu_cores > 2 else 0,
        push_to_hub=False  # Set to True if using Hugging Face Model Hub
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    multiprocessing.set_start_method('spawn', force=True)
    argsparams = parse_arguments()
    train_mlm(argsparams)
