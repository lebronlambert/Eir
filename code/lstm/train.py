# From: https://github.com/MaximumEntropy/Seq2Seq-PyTorch
"""Main script to run things"""


from data_utils import read_nmt_data, get_minibatch, get_minibatch_trg, read_config, hyperparam_string
from model import Seq2Seq
import math
import numpy as np
import logging
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument(
  "--config",
  help="path to json config",
  default = "config_eir.json"
)
args = parser.parse_args()
config_file_path = args.config
config = read_config(config_file_path)

experiment_name = hyperparam_string(config)
save_dir = config['data']['save_dir']
load_dir = config['data']['load_dir']
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  filename='log/%s' % (experiment_name),
  filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


print 'Reading data ...'

src, trg = read_nmt_data(
  src=config['data']['src'],
  config=config,
  trg=config['data']['trg']
)

src_test, trg_test = read_nmt_data(
  src=config['data']['test_src'],
  config=config,
  trg=config['data']['test_trg']
)

batch_size = config['data']['batch_size']
max_length = config['data']['max_src_length']
src_vocab_size = len(src['word2id'])

logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['data']['task']))
logging.info('Model : %s ' % (config['model']['seq2seq']))
logging.info('Source Word Embedding Dim  : %s' % (config['model']['dim_word_src']))
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
logging.info('Batch Size : %d ' % (config['data']['batch_size']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))

logging.info('Found %d words in src ' % (src_vocab_size))

loss_criterion = nn.CrossEntropyLoss().cuda()

if config['model']['seq2seq'] == 'vanilla':
  model = Seq2Seq(
    src_emb_dim=config['model']['dim_word_src'],
    src_vocab_size=src_vocab_size,
    src_hidden_dim=config['model']['dim'],
    batch_size=batch_size,
    bidirectional=config['model']['bidirectional'],
    pad_token_src=src['word2id']['<pad>'],
    nlayers=config['model']['n_layers_src'],
    dropout=0.,
  ).cuda()

if load_dir:
    model.load_state_dict(torch.load(
      open(load_dir)
    ))

# __TODO__ Make this more flexible for other learning methods.
if config['training']['optimizer'] == 'adam':
  lr = config['training']['lrate']
  optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'adadelta':
  optimizer = optim.Adadelta(model.parameters())
elif config['training']['optimizer'] == 'sgd':
  lr = config['training']['lrate']
  optimizer = optim.SGD(model.parameters(), lr=lr)
else:
  raise NotImplementedError("Learning method not recommend for task")

for i in xrange(100):

  error = 0
  for j in xrange(0, len(src['data']), batch_size):

    input_lines_src, _, lens_src, mask_src = get_minibatch(
      src['data'], src['word2id'], j,
      batch_size, max_length, add_start=True, add_end=True
    )
    output_lines_trg = get_minibatch_trg(
      trg['data'], j, batch_size
    )

    optimizer.zero_grad()
    outputs = model(input_lines_src)
    _, indices = torch.max(outputs, 1)

    tmp = 1.0*sum(output_lines_trg != indices).data.cpu().numpy()[0]
    error += tmp
  error = 1.0*error/len(src['data'])
  print 'epoch',i,error


  for j in xrange(0, len(src['data']), batch_size):
    input_lines_src, _, lens_src, mask_src = get_minibatch(
      src['data'], src['word2id'], j,
      batch_size, max_length, add_start=True, add_end=True
    )
    output_lines_trg = get_minibatch_trg(
      trg['data'], j, batch_size
    )

    optimizer.zero_grad()
    outputs = model(input_lines_src)

    loss = loss_criterion(
      outputs,
      output_lines_trg
    )

    loss.backward()
    optimizer.step()


  torch.save(
    model.state_dict(),
    open(os.path.join(
      save_dir,
      experiment_name + '__epoch_%d' % (i) + '.model'), 'wb'
    )
  )
