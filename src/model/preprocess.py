import re
import unicodedata
import string
from abc import ABC
from string import punctuation
from pandarallel import pandarallel

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer

pandarallel.initialize()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

FILE_PATH = "../../data/"
FILE_LIST = ["Train.csv", "Test.csv", "Valid.csv"]

for file in FILE_LIST:
    file_content = pd.read_csv(FILE_PATH + file)
    # print(file_content)
    break


class PreprocessData:
    """
    cleaning data by remove off the punctuations, stop words and other symbols

    input: (dataframe) raw text and label file
    :return (array) x and y represent cleaned text and corresponding label
    """

    def __init__(self, files):
        self.files = files

    @staticmethod
    def __get_punctuation():
        pun = set()

        for cp in range(17 * 65536):
            char = chr(cp)
            if ((33 <= cp <= 47) or (58 <= cp <= 64) or
                    (91 <= cp <= 96) or (123 <= cp <= 126)):
                pun.add(char)
            cat = unicodedata.category(char)
            if cat.startswith("P"):
                pun.add(char)
        for i in string.punctuation:
            pun.add(i)

        return pun

    @staticmethod
    def __preprocess_english(content):
        train_data = []

        for word in content:
            word = re.sub(r'[{}]+'.format(punctuation), ' ', word)
            train_data.append(word)

        return train_data

    def __re_text(self, text):
        puns = self.__get_punctuation()
        sent = []
        for line in text:
            data = re.sub('[0-9]', '', line)
            words = data.split() + ['']
            word_lst = []
            for word in words:
                if word not in puns:
                    word_lst.append(word)
            word_str = ' '.join(word_lst)
            sent.append(word_str)

        return sent

    def __read_file(self):
        # delete redundant spaces
        self.files["text"] = self.files["text"].parallel_apply(lambda x: re.sub(" +", " ", x))

        file_dict = {"text": self.files["text"].to_numpy(), "label": self.files["label"].to_numpy()}
        file_dict["text"] = self.__preprocess_english(self.__re_text(file_dict["text"]))

        return file_dict

    def __split_data(self):
        return self.__read_file()["text"], self.__read_file()["label"]

    def __call__(self, *args, **kwargs):
        return self.__split_data()


preprocess = PreprocessData(file_content)
preprocessed_X, preprocessed_y = preprocess()
print(len(preprocessed_X))


class Vocabulary:
    """
    tokenize and create vocabulary
    :return index2word word2index
    """

    def __init__(self, threshold, max_size):
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word2index = {k: j for j, k in self.index2word.items()}

        self.threshold = threshold  # the minimum times a word must occur in corpus to be treated in vocab
        self.max_size = max_size  # max source vocab size.

    def __len__(self):
        return len(self.index2word)

    @staticmethod
    def __tokenizer(text):
        return [token.lower().strip() for token in text.split(' ')]

    def __build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.__tokenizer(sentence):
                if word not in frequencies.keys():
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

        frequencies = {k: v for k, v in frequencies.items() if v > self.threshold}

        frequencies = dict(
            sorted(frequencies.items(), key=lambda x: -x[1])[:int(self.max_size - idx)])

        for word in frequencies.keys():
            self.word2index[word] = idx
            self.index2word[idx] = word
            idx += 1

        return self.word2index, self.index2word

    def word_index(self, text):
        tokenized_text = self.__tokenizer(text)
        number_text = []

        for token in tokenized_text:
            if token in self.word2index.keys():
                number_text.append(self.word2index[token])
            else:
                number_text.append(self.word2index['<UNK>'])

        return number_text

    def __call__(self, sent_lst,  *args, **kwargs):
        return self.__build_vocabulary(sent_lst)


class MyDataset(Dataset, ABC):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __getitem__(self, item):
        single_text = self.text[item]
        label = self.labels[item]

        tokenized = tokenizer(single_text, padding='max_length', truncation=True, max_length=320, return_tensors='pt')
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        token_type_ids = tokenized['token_type_ids'].squeeze(0)
        return input_ids, attention_mask, token_type_ids, label

    def __len__(self):
        return len(self.text)


dataset = MyDataset(preprocessed_X, preprocessed_y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for idx, batch in enumerate(dataloader):
    input_ids, attention_mask, token_type_ids, label = batch[0], batch[1], \
                                                   batch[2], batch[3]

    """
    input_ids: [batch_size, max_sequence_length]
    attention_mask: [batch_size, max_sequence_length]
    token_type_ids: [batch_size, max_sequence_length]
    label: [batch_size]
    """

    break
