from __future__ import division, unicode_literals

import numpy as np
import json
from multiprocessing import Pool
import subprocess
import glob,os
import random
import math

import re
import pickle


## Part 1: Get the TF-IDF for each word in each document in descending order.
# http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/
from textblob import TextBlob as tb

def tf(word, blob):
  return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
  return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
  return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
  return tf(word, blob) * idf(word, bloblist)

def f(idx):
  with open('data/trait_wiki-1') as data_file:
    data = json.load(data_file)

  set_trait = list(data.keys())
  bloblist = [tb(data[trait]) for trait in set_trait]

  trait2tfidf = {}

  blob = bloblist[idx]

  scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
  sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
  trait2tfidf[set_trait[idx]] = sorted_words

  with open('data/trait/trait_extract_'+str(idx), 'w') as data_file:
    json.dump(trait2tfidf, data_file)
  print("{}: {}".format(idx, set_trait[idx]))

with open('data/trait_wiki') as data_file:
  data = json.load(data_file)
set_trait = list(data.keys())
setidx = list(range(len(set_trait)))

pool = Pool(processes=62)
pool.map(f,setidx)


## Part 2: Aggregate the words with top TF-IDF for each phenotype.

topw = 20
trait2words = {}
for idx in range(7188):
  with open('data/trait/trait_extract_'+str(idx)) as data_file:
    data = json.load(data_file)
    trait2words[data.keys()[0]] = [ it[0] for it in data[data.keys()[0]][:min(topw,len(data[data.keys()[0]])-5)] ]

with open('data/trait2words'+str(topw), 'w') as data_file:
  json.dump(trait2words, data_file)


## Part 3: substitute the phenotype words in abstract with 'PHENOTYPE'

topw = 20

with open('data/trait2words'+str(topw)) as data_file:
  trait2words = json.load(data_file)

data = np.load('data/ground_truth4000_articles')

for k in data.keys():
  gene, trait = k
  # trait not necessarily found in wikipedia
  if trait not in trait2words.keys(): continue
  list_info = data[k]
  for idx, info in enumerate(list_info):
    # substitute abstract of each entry of infomation
    temp = info['abstract']
    for word in trait2words[trait]:
      temp = re.sub(word, 'PHENOTYPE', temp)
    splitemp = []
    temp = temp.split('. ')
    for line in temp:
      if ('GENE' not in line) and ('PHENOTYPE' not in line): continue
      line = line.strip().split()
      for idxw, word in enumerate(line):
        if 'GENE' in word:
          line[idxw] = 'GENE'
        elif 'PHENOTYPE' in word:
          line[idxw] = 'PHENOTYPE'
      line = ' '.join(line)
      splitemp.append(line)

    data[k][idx]['abstract'] = '. '.join(splitemp)

with open('data/ground_truth4000_articles_PHENOTYPE'+str(topw), 'wb') as outfile:
  pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)


## Part 4: Prepare the train and test dataset.

topw = 20
prop_train = 0.6

data = np.load('data/ground_truth4000_articles_PHENOTYPE'+str(topw))
pstv = np.load('data/ground_truth4000')

text = []
label = []

for k in data.keys():
  gene, trait = k
  list_info = data[k]

  for info in list_info:
    text.append(info['abstract'])
    if info['PMID'] in pstv[k]:
      label.append(1)
    else:
      label.append(0)

num = len(label)

order = range(num)
random.seed(666)
random.shuffle(order)

text = [text[idx] for idx in order]
label = [label[idx] for idx in order]

num_train = int(num*prop_train)

text_train = text[0:num_train]
text_test = text[num_train:]
label_train = label[0:num_train]
label_test = label[num_train:]

with open('data/train_text'+str(topw), 'wb') as outfile:
  pickle.dump(text_train, outfile, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/test_text'+str(topw), 'wb') as outfile:
  pickle.dump(text_test, outfile, protocol=pickle.HIGHEST_PROTOCOL)
np.save('data/train_label'+str(topw),label_train)
np.save('data/test_label'+str(topw),label_test)

