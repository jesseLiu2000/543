from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
import torch.nn as nn

from ignite.contrib.handlers import PiecewiseLinear
from ignite.handlers import ModelCheckpoint
from ignite.engine import Engine
from ignite.engine import Events
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Accuracy
from ignite.handlers import EarlyStopping


class Transformer(nn.Module):
    """
    Transformer model based on the bert-base-uncased weight with ignite engine to speed up theh training process
    """

    @staticmethod
    def load_dataset():
        return load_dataset("imdb")

    @staticmethod
    def tokenize_function(examples):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    def __init__(self, optimizer):
        super(Transformer, self).__init__()
        self.optimizer_type = optimizer
        self.lr = 5e-5
        self.num_labels = 2
        self.num_epochs = 1
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.train_dataloader, self.eval_dataloader = self.create_dataloader()
        self.num_training_steps = self.num_epochs * len(self.train_dataloader)
        self.milestones_values = [
            (0, self.lr),
            (self.num_training_steps, 0.0),
        ]
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                        num_labels=self.num_labels).to(self.device)

        assert self.optimizer_type in ["Adam", "AdamW", "SGD"]

        if self.optimizer_type == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "AdamW":
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "SGD":
            self.optimizer = SGD(self.model.parameters(), lr=self.lr)

        self.lr_scheduler = PiecewiseLinear(
            self.optimizer, param_name="lr", milestones_values=self.milestones_values
        )

    def create_dataloader(self):
        small_train_dataset, small_eval_dataset = self.create_dataset()
        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
        eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
        return train_dataloader, eval_dataloader

    def create_dataset(self):
        raw_datasets = self.load_dataset()
        tokenized_datasets = raw_datasets.map(self.tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        small_train_dataset = tokenized_datasets["train"].shuffle().select(range(5000))
        small_eval_dataset = tokenized_datasets["test"].shuffle().select(range(5000))

        return small_train_dataset, small_eval_dataset

    def forward(self):

        def train_step(engine, batch):
            self.model.train()

            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss

        trainer = Engine(train_step)
        trainer.add_event_handler(Events.ITERATION_STARTED, self.lr_scheduler)
        pbar = ProgressBar()
        pbar.attach(trainer)
        pbar.attach(trainer, output_transform=lambda x: {'loss': x})

        def evaluate_step(engine, batch):
            self.model.eval()

            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            return {'y_pred': predictions, 'y': batch["labels"]}

        train_evaluator = Engine(evaluate_step)
        validation_evaluator = Engine(evaluate_step)
        Accuracy().attach(train_evaluator, 'accuracy')
        Accuracy().attach(validation_evaluator, 'accuracy')

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            train_evaluator.run(self.train_dataloader)
            metrics = train_evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            print(f"Training Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}")

        def log_validation_results(engine):
            validation_evaluator.run(self.eval_dataloader)
            metrics = validation_evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            print(f"Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}")

        def score_function(engine):
            val_accuracy = engine.state.metrics['accuracy']
            return val_accuracy

        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
        handler = EarlyStopping(patience=2, score_function=score_function, trainer=trainer)
        validation_evaluator.add_event_handler(Events.COMPLETED, handler)
        checkpointer = ModelCheckpoint(dirname='models', filename_prefix='bert-base-cased', n_saved=2, create_dir=True)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': self.model})
        trainer.run(self.train_dataloader, max_epochs=self.num_epochs)


if __name__ == '__main__':
    optimizer = sys.argv[1]
    transformer = Transformer(str(optimizer))
    transformer()
