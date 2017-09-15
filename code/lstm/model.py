# main function to train the bi-lstm model.

# From:https://raw.githubusercontent.com/MaximumEntropy/Seq2Seq-PyTorch/master/model.py
"""Sequence to Sequence models."""
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np

class Seq2Seq(nn.Module):
  """Container module with an encoder, deocder, embeddings."""

  def __init__(
    self,
    src_emb_dim,
    src_vocab_size,
    src_hidden_dim,
    batch_size,
    pad_token_src,
    bidirectional=True,
    nlayers=2,
    dropout=0.,
  ):
    """Initialize model."""
    super(Seq2Seq, self).__init__()
    self.src_vocab_size = src_vocab_size
    self.src_emb_dim = src_emb_dim
    self.src_hidden_dim = src_hidden_dim
    self.batch_size = batch_size
    self.bidirectional = bidirectional
    self.nlayers = nlayers
    self.dropout = dropout
    self.num_directions = 2 if bidirectional else 1
    self.pad_token_src = pad_token_src
    self.src_hidden_dim = src_hidden_dim // 2 \
      if self.bidirectional else src_hidden_dim

    self.src_embedding = nn.Embedding(
      src_vocab_size,
      src_emb_dim,
      self.pad_token_src
    )

    self.encoder = nn.LSTM(
      src_emb_dim,
      self.src_hidden_dim,
      nlayers,
      bidirectional=bidirectional,
      batch_first=True,
      dropout=self.dropout
    )

    # bidirectional
    self.fc1 = nn.Linear(in_features=src_hidden_dim,
                     out_features=2,
                     bias=True)

    self.init_weights()

  #TODO:
  def init_weights(self):
    """Initialize weights."""
    initrange = 0.1
    self.src_embedding.weight.data.uniform_(-initrange, initrange)

  def get_state(self, input):
    """Get cell states and hidden states."""
    batch_size = input.size(0) \
      if self.encoder.batch_first else input.size(1)
    h0_encoder = Variable(torch.zeros(
      self.encoder.num_layers * self.num_directions,
      batch_size,
      self.src_hidden_dim
    ))
    c0_encoder = Variable(torch.zeros(
      self.encoder.num_layers * self.num_directions,
      batch_size,
      self.src_hidden_dim
    ))

    return h0_encoder.cuda(), c0_encoder.cuda()

  def forward(self, input_src):
    """Propogate input through the network."""
    src_emb = self.src_embedding(input_src)

    self.h0_encoder, self.c0_encoder = self.get_state(input_src)

    src_h, (src_h_t, src_c_t) = self.encoder(
      src_emb, (self.h0_encoder, self.c0_encoder)
    )

    if self.bidirectional:
      h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
      c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
    else:
      h_t = src_h_t[-1]
      c_t = src_c_t[-1]

    return F.softmax(self.fc1(h_t))

