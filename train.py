import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_cuda_device():
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return "cpu"


class MultiHeadClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels_list):
        super().__init__()
        self.bert = base_model
        self.num_labels_list = num_labels_list
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(self.bert.config.hidden_size, num_labels)
                for num_labels in num_labels_list
            ]
        )

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.pooler_output

        logits = [classifier(pooled_output) for classifier in self.classifiers]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = sum(
                loss_fct(logit, label) for logit, label in zip(logits, labels.t())
            )

        return (loss, logits) if loss is not None else logits


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(f"Logits shape: {[logit.shape for logit in logits]}")
    print(f"Labels shape: {labels.shape}")

    predictions = [np.argmax(logit, axis=1) for logit in logits]
    print(f"Predictions shape: {[pred.shape for pred in predictions]}")

    results = {}
    for i, (preds, labs) in enumerate(zip(predictions, labels.T)):
        precision = precision_score(labs, preds, average="macro", zero_division=0)
        recall = recall_score(labs, preds, average="macro", zero_division=0)
        f1 = f1_score(labs, preds, average="macro", zero_division=0)
        accuracy = accuracy_score(labs, preds)

        results[f"{score_columns[i]}_precision"] = precision
        results[f"{score_columns[i]}_recall"] = recall
        results[f"{score_columns[i]}_f1"] = f1
        results[f"{score_columns[i]}_accuracy"] = accuracy

        print(f"Classification Report for {score_columns[i]}:")
        print(classification_report(labs, preds, zero_division=0))
        print(f"Confusion Matrix for {score_columns[i]}:")
        print(confusion_matrix(labs, preds))

    return results


def main(args):
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")

    device = get_available_cuda_device()
    logger.info(f"Using device: {device}")

    wandb.init(project="code_quality_assessment", name=args.run_name)

    global score_columns
    score_columns = [
        "overall_score",
        "functionality_score",
        "readability_score",
        "efficiency_score",
        "maintainability_score",
        "error_handling_score",
    ]

    dataset = load_dataset(
        args.dataset_name,
        split="train",
        cache_dir=args.cache_dir,
    )

    # Convert scores to integers and clip to range 0-5
    for column in score_columns:
        dataset = dataset.map(
            lambda x: {column: min(max(int(x[column]), 0), 5)}, num_proc=8
        )

    # Split the dataset
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    model = MultiHeadClassificationModel(
        AutoModel.from_pretrained(args.base_model_name),
        num_labels_list=[6] * len(score_columns),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    def preprocess(examples):
        text = examples["original_code"]
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=512)

        labels = [
            [int(examples[column][i]) for column in score_columns]
            for i in range(len(text))
        ]

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

    # Process the datasets
    train_dataset = dataset["train"].map(
        preprocess, batched=True, remove_columns=dataset["train"].column_names
    )
    eval_dataset = dataset["test"].map(
        preprocess, batched=True, remove_columns=dataset["test"].column_names
    )

    # Set the format to PyTorch tensors
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        hub_model_id=args.output_model_name,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=5,
        learning_rate=3e-5,
        num_train_epochs=3,
        seed=42,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="overall_score_f1",
        greater_is_better=True,
        report_to="wandb",
    )

    if torch.cuda.is_available():
        training_args.bf16 = True
    else:
        print("CUDA is not available. Training will proceed on CPU.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="jinaai/jina-embeddings-v2-base-code"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="amazingvince/the-stack-smol-xs-scored-and-annotated",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="code_score/jina_embeddings_v2_base_code_multi_regression",
    )
    parser.add_argument(
        "--output_model_name",
        type=str,
        default="amazingvince/jina_embeddings_v2_base_code_multi_regression",
    )
    parser.add_argument("--run_name", type=str, default="code_quality_assessment_run")
    parser.add_argument("--cache_dir", type=str, default="~/cache/")
    args = parser.parse_args()

    main(args)
