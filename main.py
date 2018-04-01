"""
Main script for training
"""

from data_utils import load_dataset, load_dictionaries, get_minibatch
from model import Seq2RelNet
import numpy as np
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

path_config = 'config.json'
config = json.load(open(path_config, 'r'))

config['training']['is_cuda'] = config['training']['is_cuda'] and torch.cuda.is_available()
is_cuda = config['training']['is_cuda']
torch.manual_seed(config['training']['torch_seed'])
if is_cuda:
  print 'Using cuda...'
  torch.cuda.manual_seed(config['training']['torch_seed'])
else:
  print 'NOT using cuda...'

for trial in range(100):
  if trial == 99:
    print 'Remove the directory please'
    break
  if os.path.exists('output_trial_'+str(trial)):
    continue
  config['data']['path_output'] = 'output_trial_'+str(trial)+'/'
  os.makedirs(config['data']['path_output'])
  json.dump(config,open(config['data']['path_output']+'params.txt','w'))
  break
path_file_output = config['data']['path_output']+'output.txt'


train_set, test_set = load_dataset(config)
concept2idx, typ2idx = load_dictionaries(train_set['samples'], config)


model = Seq2RelNet(padding_idx = concept2idx['SELF_DEFINED_PAD'],
  vocab_size_concept = len(concept2idx),
  dim_emb_concept = config['model']['dim_emb_concept'],
  dim_typ = config['model']['vocab_size_typ'] + 1,
  dim_hidden = config['model']['dim_hidden'],
  bidirectional = config['model']['bidirectional'],
  num_layers = config['model']['num_layers'],
  dropout = 0.,
  is_cuda = is_cuda)
if is_cuda:
  model.cuda()


if config['training']['starting_trial'] != -1:
  path_model_param = 'output_trial_'+str(config['training']['starting_trial']) \
    +'/model_param_epoch_'+str(config['training']['starting_epoch']) \
    +'_index_'+str(config['training']['starting_index'])
  if not os.path.exists(path_model_param):
    print 'Error: path '+path_model_param+' does not exist'
  else:
    print 'Using model params from '+path_model_param
    model.load_state_dict(torch.load(path_model_param))
else:
  print 'Start training from scratch'

if is_cuda:
  loss_criterion = nn.CrossEntropyLoss().cuda()
else:
  loss_criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=config['training']['lrate'])

epochs = config['training']['epochs']
batch_size = config['data']['batch_size']
test_batch_size = config['data']['test_batch_size']
max_length = config['data']['max_length']


for epoch in xrange(epochs):
  sum_error, sum_loss, count_sample = 0, 0, 0
  for index in xrange(0, len(train_set['labels']), batch_size):
    batch_data = get_minibatch(train_set, concept2idx, typ2idx, batch_size, max_length, index)
    if is_cuda:
      outputs = model(batch_data['batch_gene_disease'].cuda(),
                      batch_data['batch_typ'].cuda(),
                      batch_data['batch_concept'].cuda())
      labels = batch_data['batch_label'].cuda()
    else:
      outputs = model(batch_data['batch_gene_disease'],
                      batch_data['batch_typ'],
                      batch_data['batch_concept'])
      labels = batch_data['batch_label']
    optimizer.zero_grad()
    loss = loss_criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    sum_loss += len(labels) * loss.data[0]
    _, indices = torch.max(outputs, 1)
    sum_error += sum(labels != indices).data[0]
    count_sample += len(labels)
    # save model parameters periodically
    if index/batch_size % 500 == 0:
      torch.save(model.state_dict(), config['data']['path_output']+'model_param_epoch_'+str(epoch)+'_index_'+str(index))
    # report the training/test loss/error periodically
    if index/batch_size % 10 == 0:
      train_error, train_loss = 100.0*sum_error/count_sample, 1.0*sum_loss/count_sample
      sum_error, sum_loss, count_sample = 0, 0, 0
      for index_test in xrange(0, 1000, test_batch_size):
        batch_data = get_minibatch(test_set, concept2idx, typ2idx, test_batch_size, max_length, index_test)
        if is_cuda:
          outputs = model(batch_data['batch_gene_disease'].cuda(),
                          batch_data['batch_typ'].cuda(),
                          batch_data['batch_concept'].cuda())
          labels = batch_data['batch_label'].cuda()
        else:
          outputs = model(batch_data['batch_gene_disease'],
                          batch_data['batch_typ'],
                          batch_data['batch_concept'])
          labels = batch_data['batch_label']
        _, indices = torch.max(outputs, 1)
        sum_error += sum(labels != indices).data[0]
        count_sample += len(labels)
        loss = loss_criterion(outputs,labels)
        sum_loss += len(labels) * loss.data[0]
      test_error, test_loss = 100.0*sum_error/count_sample, 1.0*sum_loss/count_sample
      print '[%d,%d],TrnLoss:%.3f,TstLoss:%.3f,TrnErr:%.2f,TstErr:%.2f'%(epoch,index, train_loss, test_loss, train_error, test_error)
      with open(path_file_output, 'a') as file_output:
        file_output.write(
            '[%d,%d],TrnLoss:%.3f,TstLoss:%.3f,TrnErr:%.2f,TstErr:%.2f\n' \
            %(epoch,index, train_loss, test_loss, train_error, test_error)
            )
      sum_error, sum_loss, count_sample = 0, 0, 0
