import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data
from torchtext.legacy import datasets
from torch.optim import Adam, AdamW, SGD

import random
import time
import sys


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        return self.fc(pooled)


class FastTextModule(nn.Module):

    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    @staticmethod
    def generate_bigrams(x):
        n_grams = set(zip(*[x[i:] for i in range(2)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    @staticmethod
    def binary_accuracy(preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def __init__(self, FastText, optimizer_type):
        super(FastTextModule, self).__init__()
        self.BATCH_SIZE = 64
        self.lr = 1e-4
        self.N_EPOCHS = 10
        self.best_valid_loss = float('inf')
        self.optimizer_type = optimizer_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm',
                               preprocessing=self.generate_bigrams)
        self.LABEL = data.LabelField(dtype=torch.float)
        self.train_valid_data, self.test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)
        self.train_data, self.valid_data = self.train_valid_data.split(random_state=random.seed(self.SEED))
        self.MAX_VOCAB_SIZE = 25000
        self.TEXT.build_vocab(self.train_data, max_size=self.MAX_VOCAB_SIZE, vectors="glove.6B.100d",
                              unk_init=torch.Tensor.normal_)
        self.LABEL.build_vocab(self.train_data)

        self.INPUT_DIM = len(self.TEXT.vocab)
        self.EMBEDDING_DIM = 100
        self.OUTPUT_DIM = 1
        self.PAD_IDX = self.TEXT.vocab.stoi[self.TEXT.pad_token]

        self.train_iterator, self.valid_iterator, self.test_iterator = self.build_iterator()
        self.model = FastText(self.INPUT_DIM, self.EMBEDDING_DIM, self.OUTPUT_DIM, self.PAD_IDX)

        assert self.optimizer_type in ["Adam", "AdamW", "SGD"]

        if self.optimizer_type == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "AdamW":
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "SGD":
            self.optimizer = SGD(self.model.parameters(), lr=self.lr)

    def build_iterator(self):
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=self.BATCH_SIZE,
            device=self.device)

        return train_iterator, valid_iterator, test_iterator

    def train_step(self, model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.train()
        for batch in iterator:
            optimizer.zero_grad()
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = self.binary_accuracy(predictions, batch.label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate_step(self, model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.eval()
        with torch.no_grad():
            for batch in iterator:
                predictions = model(batch.text).squeeze(1)
                loss = criterion(predictions, batch.label)
                acc = self.binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def forward(self):
        output = self.generate_bigrams(['This', 'film', 'is', 'terrible'])
        print("test bigram module----\n", output)

        print(f'The model has {self.count_parameters(self.model):,} trainable parameters')

        pretrained_embeddings = self.TEXT.vocab.vectors
        self.model.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = self.TEXT.vocab.stoi[self.TEXT.unk_token]

        self.model.embedding.weight.data[UNK_IDX] = torch.zeros(self.EMBEDDING_DIM)
        self.model.embedding.weight.data[self.PAD_IDX] = torch.zeros(self.EMBEDDING_DIM)

        criterion = nn.BCEWithLogitsLoss()

        self.model = self.model.to(self.device)
        criterion = criterion.to(self.device)

        for epoch in range(self.N_EPOCHS):
            start_time = time.time()

            train_loss, train_acc = self.train_step(self.model, self.train_iterator, self.optimizer, criterion)
            valid_loss, valid_acc = self.evaluate_step(self.model, self.valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < self.best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'fasttext-model.pt')

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        self.model.load_state_dict(torch.load('fasttext-model.pt'))

        test_loss, test_acc = self.evaluate_step(self.model, self.test_iterator, criterion)

        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


if __name__ == '__main__':
    optimizer = sys.argv[1]
    fasttext = FastTextModule(FastText, str(optimizer))
    fasttext()












