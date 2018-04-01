import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Seq2RelNet(nn.Module):
  '''
  Seq2RelNet classify whether the causality relationship exists in the input
  abstract (Already preprocessed by MetaMap and tokenized).
  '''
  def __init__(self,padding_idx,vocab_size_concept,dim_emb_concept,dim_typ,dim_hidden,bidirectional = True,num_layers = 2,dropout = 0.,is_cuda = True):
    super(Seq2RelNet, self).__init__()
    self.padding_idx = padding_idx
    self.vocab_size_concept = vocab_size_concept
    self.dim_emb_concept = dim_emb_concept
    self.dim_typ = dim_typ
    self.dim_hidden = dim_hidden
    self.bidirectional = bidirectional
    self.num_layers = num_layers
    self.dropout = dropout
    self.num_directions = 2 if bidirectional else 1
    if self.bidirectional:
      self.dim_hidden = dim_hidden // 2
    else:
      self.dim_hidden = dim_hidden
    self.is_cuda = is_cuda
    self.concept_embedding = nn.Embedding(self.vocab_size_concept,self.dim_emb_concept,self.padding_idx)
    self.encoder = nn.LSTM(
      2 + self.dim_typ + self.dim_emb_concept,
      self.dim_hidden,
      self.num_layers,
      bidirectional = bidirectional,
      batch_first = True,
      dropout = self.dropout)
    self.fc1 = nn.Linear(in_features = dim_hidden,
                     out_features = 2,
                     bias=True)
    self.init_weights()

  def init_weights(self):
    scaling_concept_emb = 0.1
    self.concept_embedding.weight.data.uniform_(-scaling_concept_emb, scaling_concept_emb)

  def get_state(self, batch_concept):
    '''
    Get zero cell states and hidden states torch Variables for initialization
    each time before training the LSTM.
    '''
    batch_size = batch_concept.size(0)
    h0_encoder = Variable(torch.zeros(
      self.encoder.num_layers * self.num_directions,
      batch_size,
      self.dim_hidden
    ))
    c0_encoder = Variable(torch.zeros(
      self.encoder.num_layers * self.num_directions,
      batch_size,
      self.dim_hidden
    ))
    if self.is_cuda:
      return h0_encoder.cuda(), c0_encoder.cuda()
    else:
      return h0_encoder, c0_encoder

  def forward(self, batch_gene_disease, batch_typ, batch_concept):
    # concept -> concept embedding
    concept_emb = self.concept_embedding(batch_concept)
    # concatenate: is_gene_disease 0/1 tensors, is_semtype 0/1 tensors, concept embedding
    gene_disease_typ_concept_emb = torch.cat( (0.1*batch_gene_disease, 0.1*batch_typ, concept_emb), 2 )
    self.h0_encoder, self.c0_encoder = self.get_state(batch_concept)
    # feed the concatenated embedding into LSTM
    self.src_h,(self.src_h_t, self.src_c_t) = self.encoder(gene_disease_typ_concept_emb,
           (self.h0_encoder, self.c0_encoder) )
    # get the hidden layer value at the last position
    if self.bidirectional:
      self.h_t = torch.cat((self.src_h_t[-1], self.src_h_t[-2]), 1)
      c_t = torch.cat((self.src_c_t[-1], self.src_c_t[-2]), 1)
    else:
      self.h_t = self.src_h_t[-1]
      c_t = self.src_c_t[-1]
    # feature feed into a simple logistic regression classifier
    return F.softmax(self.fc1(self.h_t))
