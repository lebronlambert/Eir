'''
Demo of how to use the model.py and trained parameters for test.
'''

from data_utils import load_dataset, load_dictionaries, get_minibatch
from model import Seq2RelNet
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


# you may want me to help you prepare the dataset...
is_cuda = torch.cuda.is_available()
if is_cuda:
  print 'Using cuda...'
else:
  print 'NOT using cuda...'

with open('input/test_set','r') as f:
  test_set = json.load(f)


with open('input/concept2idx','r') as f:
  concept2idx = json.load(f)
with open('input/typ2idx','r') as f:
  typ2idx = json.load(f)

model = Seq2RelNet(padding_idx = concept2idx['SELF_DEFINED_PAD'],
  vocab_size_concept = len(concept2idx),
  dim_emb_concept = 512,
  dim_typ = len(typ2idx),
  dim_hidden = 1000,
  bidirectional = True,
  num_layers = 2,
  dropout = 0.,
  is_cuda = is_cuda)
if is_cuda:
  model.cuda()

# load trained model parameters
model.load_state_dict(torch.load('output_trial_5/model_param_epoch_2_index_96000'))

if is_cuda:
  loss_criterion = nn.CrossEntropyLoss().cuda()
else:
  loss_criterion = nn.CrossEntropyLoss()

test_batch_size = 64
max_length = 300

sum_error, sum_loss, count_sample = 0, 0, 0
for index_test in xrange(0, 10000, test_batch_size):
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
print 'Test Loss:%.2f,Test Error:%.2f' % (test_loss, test_error)
