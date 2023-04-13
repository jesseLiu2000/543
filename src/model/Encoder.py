import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math

from transformers import BertModel, BertConfig, BertTokenizer
from transformers import logging
import torch.optim as optim

import preprocess

logging.set_verbosity_error()

embedding_size = 768
output_size = 2
d_ff = 1024  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 1  # number of Encoder of Decoder Layer
n_heads = 1
max_sequence_length = 320


def get_attention_pad_mask(seq_q, seq_k):
    """

    :param seq_q: input_ids
    :param seq_k: input_ids
    :var pad_attention_mask: [batch_size, 1, sequence_length]
    :return: [batch_size, sequence_length, sequence_length]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attention_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attention_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attention_mask):
        """
        ScaledDotProductAttention
        :param Q: [batch_size, num_heads, seq_length, embedding_size]
        :param K: [batch_size, num_heads, seq_length, embedding_size]
        :param V: [batch_size, num_heads, seq_length, embedding_size]
        :param attention_mask: [batch_size, num_heads, seq_length, seq_length]
        :var: scores: [batch_size, num_heads, seq_length, seq_length]
        :returns: attention_score: [batch_size, num_heads, seq_length, seq_length]
                  context: [batch_size, num_heads, seq_length, seq_length]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        scores.masked_fill_(attention_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attention_score = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attention_score, V)
        return context, attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(embedding_size, d_k * n_heads)
        self.W_K = nn.Linear(embedding_size, d_k * n_heads)
        self.W_V = nn.Linear(embedding_size, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, Q, K, V, attention_mask):
        """

        :param Q: [batch_size, seq_length, embedding_size]
        :param K: [batch_size, seq_length, embedding_size]
        :param V: [batch_size, seq_length, embedding_size]
        :param attention_mask: [batch_size, seq_length, seq_length]
        :var attention_mask: [batch_size, num_heads, seq_length, seq_length]
        :var context: [batch_size, num_heads, seq_length, embedding_size]
        :returns: attention_score: [batch_size, num_heads, seq_length, seq_length]
                  output: [batch_size, seq_length, embedding_size]
        """
        residual, batch_size = Q, Q.size(0)
        q_head = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_head = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_head = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attention_mask = attention_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attention_score = ScaledDotProductAttention()(q_head, k_head, v_head, attention_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attention_score


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embedding_size, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=embedding_size, kernel_size=1)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.act = nn.ReLU()

    def forward(self, encoder_inputs, encoder_outputs):
        """
        PoswiseFeedForwardNet
        :param encoder_inputs: [batch_size, seq_length, embedding_size]
        :param encoder_outputs: [batch_size, seq_length, embedding_size]
        :var cov1_outpput: [batch_size, d_ff, seq_length]
        :var cov2_outpput: [batch_size, seq_length, embedding_size]
        :return:[batch_size, seq_length, embedding_size]
        """
        residual = encoder_inputs
        cov1_outpput = self.act(self.conv1(encoder_outputs.transpose(1, 2)))
        cov2_outpput = self.conv2(cov1_outpput).transpose(1, 2)
        return self.layer_norm(cov2_outpput + residual)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        # pe: [seq_length, embedding_size]

        # pe: [seq_length, 1, embedding_size]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        input: [max_sequence_length, batch_size, embedding_size]
        """
        y = x + self.pe[:x.size(0), :]

        return self.dropout(y)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.poswise_ffn = PoswiseFeedForwardNet()

    def forward(self, encoder_inputs, encoder_attention_mask):
        """
        EncoderLayer contain multihead attention layer and poswise FF layer
        :param encoder_inputs: [batch_size, sequence_length, embedding_size]
        :param enc_attention_mask: [batch_size, sequence_length, sequence_length]
        :return:
        """
        encoder_outputs, attention_score = self.enc_self_attn(encoder_inputs, encoder_inputs, encoder_inputs, encoder_attention_mask)
        encoder_outputs = self.poswise_ffn(encoder_inputs, encoder_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return encoder_outputs, attention_score


class EmbeddingModule(nn.Module):
    def __init__(self):
        super(EmbeddingModule, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask, token_type_ids):
        word_embedding = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        return word_embedding


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.token_embedding = EmbeddingModule()
        self.position_embedding = PositionalEncoding(embedding_size)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Encoder Module
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :var: token_embedding_outputs: [batch_size, sequence_length, embedding_size]
        :var: position_embedding_outputs: [batch_size, sequence_length, embedding_size]
        :var: encoder_attention_mask: [batch_size, sequence_length, sequence_length]
        :return: encoder_outputs, encoder_self_attention
        """
        token_embedding_outputs = self.token_embedding(input_ids, attention_mask, token_type_ids)
        position_embedding_outputs = self.position_embedding(token_embedding_outputs.transpose(0, 1)).transpose(0, 1)
        encoder_attention_mask = get_attention_pad_mask(input_ids, input_ids)

        encoder_self_attention_lst = []
        for layer in self.layers:
            encoder_outputs, encoder_self_attention = layer(position_embedding_outputs, encoder_attention_mask)
            encoder_self_attention_lst.append(encoder_self_attention)
        return encoder_outputs, encoder_self_attention


class GatheredModule(nn.Module):
    def __init__(self, input_ids, attention_mask, token_type_ids):
        super(GatheredModule, self).__init__()
        self.encoder = Encoder()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.fc = nn.Linear(embedding_size, 1)
        self.projection = nn.Linear(max_sequence_length, 2, bias=False)

    def forward(self):
        encoder_output, encoder_self_attention = self.encoder(self.input_ids, self.attention_mask, self.token_type_ids)
        output = self.projection(self.fc(encoder_output).squeeze())
        return output


def get_acc(outputs, labels):
    """计算acc"""
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0]*1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num

    return acc


if __name__ == '__main__':
    epoch = 1000
    for ep in range(epoch):
        epoch_loss = 0
        epoch_acc = 0
        for idx, batch in enumerate(preprocess.dataloader):
            input_ids, attention_mask, token_type_ids, label = batch[0], batch[1], \
                                                               batch[2], batch[3]
            model = GatheredModule(input_ids, attention_mask, token_type_ids)
            encoder_output = model()

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-5)

            optimizer.zero_grad()
            loss = loss_fn(encoder_output, label)
            # print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss.data
            epoch_acc += get_acc(encoder_output, label)

            print('epoch: %d, loss: %f, acc: %f' % (ep, epoch_loss, epoch_acc))

        # if ep % 10 == 0:
        #     print('epoch: %d, loss: %f, acc: %f' % (ep, epoch_loss / 10, epoch_acc / 10))

