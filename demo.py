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

with open('input/train_set','r') as f:
  train_set = json.load(f)


def prepare_set_all(test_set):
  set_all=[]
  for j in  range(len(test_set['samples'])):
    name1= test_set['genes'][j]
    name2=test_set['diseases'][j]
    set_=[]
    for i in range(len(test_set['samples'])):
      if name1==test_set['genes'][i] and name2==test_set['diseases'][i]:
        set_.append(i)
      else:
        pass
    if len(set_)>=2 and set_ not in set_all:
      set_all.append(set_)
  with open('input/set_all', 'w') as f:
      json.dump(set_all,f)

def generate_set(test_set,number,label,name):
  import random
  set_=[]
  sum_=0.
  for i in range(len(test_set['labels'])):
    if test_set['labels'][i]==label:
        sum_+=1
        if random.random()>=number:
          set_.append(i)
  import copy
  test_set2=copy.copy(test_set)
  test_set2['samples'],test_set2['genes'], test_set2['labels'],test_set2['pmids'],test_set2['diseases']=[],[],[],[],[]
  for i in range(len(test_set['samples'])):
      if i in set_:
          pass
      else:
          test_set2['samples'].append(test_set['samples'][i])
          test_set2['genes'].append(test_set['genes'][i])
          test_set2['labels'].append(test_set['labels'][i])
          test_set2['pmids'].append(test_set['pmids'][i])
          test_set2['diseases'].append(test_set['diseases'][i])
  with open(name, 'w') as f:
      json.dump(test_set2,f)

def function_set_all_number(name1, name2):
  with open(name1, 'r') as f:
      set_all1 = json.load(f)
  sum_sum = 0.
  set_all2 = []
  for i in range(len(set_all1)):
      if len(set_all1[i]) >= 4: #11067 2.63522183067  #50239 3.85357988813
          # >=3 #3338 4.10605152786  #33494  4.78025915089   #>=4 #20171 #5.95612513014 #1132 #6.26148409894
          set_all2.append(set_all1[i])
          sum_sum += len(set_all1[i])

  # print len(set_all2)
  # print sum_sum/len(set_all2)
  with open(name2, 'w') as f:
      json.dump(set_all2, f)

# generate_set(train_set,0.2,1,'input/normal_0.2_train_set')
# print "1"
# generate_set(test_set,0.2,1,'input/normal_0.2_test_set')
# print "2"
# generate_set(train_set,-0.1,1,'input/no_1_train_set')
# print "3"
# generate_set(test_set,-0.1,1,'input/no_1_test_set')
# print "4"
# generate_set(train_set,-0.1,0,'input/no_0_train_set')
# print "5"
# generate_set(train_set,-0.1,0,'input/no_0_test_set')
# print "6"


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

