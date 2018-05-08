import os
import numpy as np
import collections
import operator
import json

import torch
import torch.nn as nn
from torch.autograd import Variable

def partition_samples(path_pos_samples, path_neg_samples, ratio_train_samples, partition_samples_seed):
  '''
  Shuffle and partition the dataset.
  '''
  with open(path_pos_samples, 'r') as f:
    pos_samples = json.load(f) #gdscts
  with open(path_neg_samples, 'r') as f:
    neg_samples = json.load(f)
  samples = pos_samples + neg_samples
  y = [1]*len(pos_samples)+[0]*len(neg_samples)
  with open('input/pos_samples_pmid', 'r') as f:
    pos_samples_pmid = json.load(f)
  with open('input/neg_samples_pmid', 'r') as f:
    neg_samples_pmid = json.load(f)
  with open('input/pos_samples_gene', 'r') as f:
    pos_samples_gene = json.load(f)
  with open('input/neg_samples_gene', 'r') as f:
    neg_samples_gene = json.load(f)
  with open('input/pos_samples_disease', 'r') as f:
    pos_samples_disease = json.load(f)
  with open('input/neg_samples_disease', 'r') as f:
    neg_samples_disease = json.load(f)
  samples_pmid = pos_samples_pmid + neg_samples_pmid
  samples_gene = pos_samples_gene + neg_samples_gene
  samples_disease = pos_samples_disease + neg_samples_disease
  arnd = np.arange(len(samples))
  np.random.seed(partition_samples_seed)
  np.random.shuffle(arnd)
  samples = [samples[arnd[idx]] for idx in range(len(samples))]
  y = [y[arnd[idx]] for idx in range(len(samples))]
  samples_pmid = [samples_pmid[arnd[idx]] for idx in range(len(samples))]
  samples_gene = [samples_gene[arnd[idx]] for idx in range(len(samples))]
  samples_disease = [samples_disease[arnd[idx]] for idx in range(len(samples))]
  num_train_samples = int(ratio_train_samples*len(samples))
  train_set = {'samples':samples[0:num_train_samples],'labels':y[0:num_train_samples],'pmids':samples_pmid[0:num_train_samples],'genes':samples_gene[0:num_train_samples],'diseases':samples_disease[0:num_train_samples]}
  test_set = {'samples':samples[num_train_samples:],'labels':y[num_train_samples:],'pmids':samples_pmid[num_train_samples:],'genes':samples_gene[num_train_samples:],'diseases':samples_disease[num_train_samples:]}
  return train_set, test_set
def load_dataset(config):
  path_train_set = config['data']['path_train_set']
  path_test_set = config['data']['path_test_set']
  path_pos_samples = config['data']['path_pos_samples']
  path_neg_samples = config['data']['path_neg_samples']
  ratio_train_samples = config['data']['ratio_train_samples']
  partition_samples_seed = config['data']['partition_samples_seed']
  if ( os.path.isfile(path_train_set) and os.path.isfile(path_test_set) ):
    print 'Dataset found, loading dataset...'
    with open(path_train_set,'r') as f:
      train_set = json.load(f)
    with open(path_test_set,'r') as f:
      test_set = json.load(f)
  else:
    print 'Dataset not found, partitioning samples...'
    train_set, test_set = partition_samples(
        path_pos_samples,
        path_neg_samples,
        ratio_train_samples,
        partition_samples_seed
        )
    print 'Saving dataset...'
    with open(path_train_set,'w') as f:
      json.dump(train_set,f)
    with open(path_test_set,'w') as f:
      json.dump(test_set,f)
  return train_set, test_set



def construct_vocab_concepts(samples, vocab_size):
  '''
  Construct dictionary for Concepts.
  '''
  concepts = list()
  for gdscts in samples:
    for gdsct in gdscts:
      gene, disease, score, concept, semtype = gdsct
      concepts.append(concept)
  concept2count = collections.Counter(concepts)
  sorted_concept_and_count = sorted(concept2count.items(),key=operator.itemgetter(1),reverse=True)
  sorted_concepts = [x[0] for x in sorted_concept_and_count[:vocab_size]]
  concept2idx = {
      'SELF_DEFINED_START': 0,
      'SELF_DEFINED_PAD': 1,
      'SELF_DEFINED_END': 2,
      'SELF_DEFINED_UNKNOWN': 3}
  idx2concept = {
      0: 'SELF_DEFINED_START',
      1: 'SELF_DEFINED_PAD',
      2: 'SELF_DEFINED_END',
      3: 'SELF_DEFINED_UNKNOWN'}
  for idx, concept in enumerate(sorted_concepts):
    concept2idx[concept] = idx + 4
    idx2concept[idx + 4] = concept
  return concept2idx, idx2concept

def construct_vocab_typs(samples, vocab_size):
  '''
  Construct dictionary for SemType.
  '''
  typs = list()
  for gdscts in samples:
    for gdsct in gdscts:
      gene, disease, score, concept, typ = gdsct
      for tp in typ:
        typs.append(tp) # for-loop is faster than '+'
  typ2count = collections.Counter(typs)
  sorted_typ_and_count = sorted(typ2count.items(),key=operator.itemgetter(1),reverse=True)
  sorted_typs = [x[0] for x in sorted_typ_and_count[:vocab_size]]
  typ2idx = {'SELF_DEFINED_UNKNOWN_TYPE': 0}
  idx2typ = {0: 'SELF_DEFINED_UNKNOWN_TYPE'}
  for idx, typ in enumerate(sorted_typs):
    typ2idx[typ] = idx + 1
    idx2typ[idx + 1] = typ
  return typ2idx, idx2typ

def load_dictionaries(samples, config):
  path_concept2idx = config['data']['path_concept2idx']
  path_typ2idx = config['data']['path_typ2idx']
  vocab_size_concept = config['model']['vocab_size_concept']
  vocab_size_typ = config['model']['vocab_size_typ']
  if ( os.path.isfile(path_concept2idx) and os.path.isfile(path_typ2idx) ):
    print 'concept2idx, typ2idx found, loading concept2idx, typ2idx...'
    with open(path_concept2idx,'r') as f:
      concept2idx = json.load(f)
    with open(path_typ2idx,'r') as f:
      typ2idx = json.load(f)
  else:
    print 'concept2idx, typ2idx not found, constructing concept & typ vocab...'
    concept2idx, _ = construct_vocab_concepts(samples, vocab_size_concept)
    typ2idx, _ = construct_vocab_typs(samples, vocab_size_typ)
    print 'Saving concept2idx, typ2idx...'
    with open(path_concept2idx,'w') as f:
      json.dump(concept2idx,f)
    with open(path_typ2idx,'w') as f:
      json.dump(typ2idx,f)
  return concept2idx, typ2idx

def get_minibatch_genes_diseases(samples, max_length):
  samples_genes = [[gdscts[0] for gdscts in sample] for sample in samples]
  samples_diseases = [[gdscts[1] for gdscts in sample] for sample in samples]
  samples_genes = [
      [0] + concepts + [0]
      for concepts in samples_genes
      ]
  samples_genes = [genes[:max_length] for genes in samples_genes]
  samples_diseases = [
      [0] + diseases + [0]
      for diseases in samples_diseases
      ]
  samples_diseases = [diseases[:max_length] for diseases in samples_diseases]
  lens = [len(genes) for genes in samples_genes]
  max_length = max(lens)
  samples_genes = [
      genes + [0] * (max_length - len(genes))
      for genes in samples_genes
      ]
  samples_diseases = [
      diseases + [0] * (max_length - len(diseases))
      for diseases in samples_diseases
      ]
  batch_gene_disease = [
      [[gene, disease] for gene, disease in zip(genes, diseases)]
      for genes, diseases in zip(samples_genes, samples_diseases)
      ]
  return batch_gene_disease

def get_minibatch_typs(samples, typ2idx, max_length):
  samples_typs = [[gdscts[4] for gdscts in sample] for sample in samples]
  samples_typs = [
      [[]]+typs+[[]]
      for typs in samples_typs
      ]
  samples_typs = [typs[:max_length] for typs in samples_typs]
  lens = [len(typs) for typs in samples_typs]
  max_length = max(lens)
  batch_typ = np.zeros((len(samples_typs),max_length, len(typ2idx)),dtype=int)
  for idx_sample, typs in enumerate(samples_typs):
    for idx_typ, typ in enumerate(typs):
      for tp in typ:
        if tp in typ2idx:
          batch_typ[idx_sample,idx_typ,typ2idx[tp]] = 1
        else:
          batch_typ[idx_sample,idx_typ,typ2idx['SELF_DEFINED_UNKNOWN_TYPE']] = 1
  batch_typ.astype(float)
  return batch_typ

def get_minibatch_concepts(samples, concept2idx, max_length):
  batch_concept = [[gdscts[3] for gdscts in sample] for sample in samples]
  batch_concept = [
      ['SELF_DEFINED_START'] + concepts + ['SELF_DEFINED_END']
      for concepts in batch_concept
      ]
  batch_concept = [concepts[:max_length] for concepts in batch_concept]
  lens = [len(concepts) for concepts in batch_concept]
  max_length = max(lens)
  batch_concept = [
      [concept2idx[concept] if concept in concept2idx else concept2idx['SELF_DEFINED_UNKNOWN'] for concept in concepts]
      + [concept2idx['SELF_DEFINED_PAD']] * (max_length - len(concepts))
      for concepts in batch_concept
      ]
  return batch_concept

def get_minibatch_labels(labels):
  return labels

def get_minibatch(dataset, concept2idx, typ2idx, batch_size, max_length, index):
  '''
  Get a minibatch of data each time.
  Change the sample into expected format.
  We process it right before training/test because it consumes a lot of space.
  '''
  batch_gene_disease = get_minibatch_genes_diseases(dataset['samples'][index:index+batch_size],
                                                  max_length)
  batch_typ = get_minibatch_typs(dataset['samples'][index:index+batch_size],
                                 typ2idx, max_length)
  batch_concept = get_minibatch_concepts(dataset['samples'][index:index+batch_size],
                                         concept2idx, max_length)
  batch_label = get_minibatch_labels(dataset['labels'][index:index+batch_size])
  batch_gene_disease = Variable(torch.FloatTensor(batch_gene_disease))
  batch_typ = Variable(torch.FloatTensor(batch_typ))
  batch_concept = Variable(torch.LongTensor(batch_concept))
  batch_label = Variable(torch.LongTensor(batch_label))
  batch_data = {'batch_gene_disease': batch_gene_disease,
                'batch_typ': batch_typ,
                'batch_concept': batch_concept,
                'batch_label': batch_label}
  return batch_data

