import os
import numpy as np
from collections import defaultdict as dd

if not os.path.exists('input'):
  os.makedirs('input')

def getMetaMap(path):
  import json
  q2c = dd(list)
  qcs = json.load(open(path, 'r'))
  k = 0
  for qc in qcs:
    k += 1
    cs, q = qc['concepts'], qc['query']
    for c in cs:
      q2c[q].append(c['CandidateCUI'])
  return q2c

d2c = getMetaMap('raw/phenotypes_ascii_output.txt') #7384/7448
g2c = getMetaMap('raw/genes_ascii_output.txt') #5126/6196



def getGeneDisease2Pmid(path):
  articles = np.load(path) # download_articles
  gd2pmid = dd(list)
  for k in articles.keys():
    gene, disease = k
    lst = articles[k]
    for ele in lst:
      pmid = ele['PMID']
      gd2pmid[(gene, disease)].append(pmid)
  return gd2pmid

if not os.path.isfile('input/gd2pmid.npy'):
  print '(GENE, disease) -> PMID...'
  path = 'raw/final_output'
  gd2pmid = getGeneDisease2Pmid(path)
  np.save('input/gd2pmid.npy',gd2pmid)



def getGeneDisease2NPmid(gd2pmid, gd2ppmid):
  gd2npmid = dd(list)
  for k in gd2pmid.keys():
    gene, disease = k
    pmid = set(gd2pmid[k])
    ppmid = set(gd2ppmid[k])

    if not ppmid.issubset(pmid):
      ppmid = ppmid & pmid
      gd2ppmid[k] = list(ppmid)
    gd2npmid[k] = list(pmid - ppmid)
  return gd2npmid

print 'Loading: (GENE, disease) -> PMID...'
gd2pmid = np.load('input/gd2pmid.npy').item()
gd2ppmid = np.load('raw/ground_truth')
gd2npmid = getGeneDisease2NPmid(gd2pmid, gd2ppmid)



def getP2SCT(trial):
  p2sct = dd(list)
  print 'Reading result'+str(trial)+'...'
  f = open('raw/textResult/result'+str(trial),'r')
  articles = f.read()
  f.close()
  articles = articles.split('----------------\n')[1:]
  for article in articles:
    article = article.strip().split('\n')
    if len(article) <= 2:
      continue # void article
    pmid = int(article[1])

    lst = article[2:]
    sct = list()
    for ele in lst:
      ele = ele.split('\t')
      sct.append( (int(ele[0]),ele[1],list(ele[2:])) )
    p2sct[pmid] = sct
  np.save('input/p2sct'+str(trial)+'.npy',p2sct)

def getP2SCTs_sub():
  from multiprocessing import Pool
  flag_exist = True
  for trial in range(8):
    if not os.path.isfile('input/p2sct'+str(trial)+'.npy'):
      flag_exist = False
      break
  if not flag_exist:
    print 'PMID -> (score, concept, type)'
    pool = Pool(processes=8)
    pool.map(getP2SCT,range(8))

def getP2SCTs():
  import json
  if not os.path.isfile('input/p2sct'):
    getP2SCTs_sub()

    print 'Merging PMID ->  [(score, concept, type)]'
    p2sct = dd(list)
    for trial in range(8):
      tmp = np.load('input/p2sct'+str(trial)+'.npy').item()
      p2sct.update(tmp)
    with open('input/p2sct', 'w') as f:
      json.dump(p2sct,f)

  print 'Loading PMID -> [(score, concept, type)]'
  with open('input/p2sct', 'r') as f:
    p2sct = json.load(f)
  return p2sct

pmid2sct = getP2SCTs()

def getSample(gd2pmid, setpmid, path):
  import random
  corpus = list()
  for gd in gd2pmid.keys():
    gene, disease = gd
    pmids = gd2pmid[gd]
    for pmid in pmids:
      # since our mapped results are not complete
      if pmid not in setpmid: continue
      corpus.append((gene,disease,pmid))
  random.Random(666).shuffle(corpus)
  np.save(path, corpus)

if not (os.path.isfile('input/pos_corpus.npy') \
        and os.path.isfile('input/neg_corpus.npy') ):
  print 'Shuffling corpus...'
  setpmid = set([int(pmid) for pmid in pmid2sct.keys()])
  getSample(gd2ppmid, setpmid, 'input/pos_corpus.npy')
  getSample(gd2npmid, setpmid, 'input/neg_corpus.npy')

print 'Loading corpus...'
pos_corpus = np.load('input/pos_corpus.npy')
neg_corpus = np.load('input/neg_corpus.npy')



# we do not take the gene/disease names that cannot even interpreted
# by MetaMap/UMLS

def replaceOntology(pmid2sct, corpus, g2c, d2c):
  num_no_gene = 0
  num_no_disease = 0
  num_no_count = 0
  samples = list() #[gdscts,...]
  samples_pmid = list()
  for gdp in corpus:
    gene, disease, pmid = gdp
    if gene not in g2c.keys():
      num_no_gene += 1
    if disease not in d2c.keys():
      num_no_disease += 1
    if (gene not in g2c.keys()) or (disease not in d2c.keys()):
      num_no_count += 1
      continue
    scts = pmid2sct[pmid]
    gdscts = list() #[[is_gene,is_disease,score,concept,[semtype,...]],...]
    gene_concepts = g2c[gene]
    disease_concepts = d2c[disease]
    flag_gene = False
    flag_disease = False
    for idx,sct in enumerate(scts):
      s, c, t = sct
      gdsct = [0,0,s,c,t]
      if c in gene_concepts:
        gdsct[0] = 1
        flag_gene = True
      if c in disease_concepts:
        gdsct[1] = 1
        flag_disease = True
      gdscts.append(gdsct)
    samples.append(gdscts)
    samples_pmid.append(pmid)
  num_sample = len(corpus)
  print 100.0*len(samples)/num_sample
  return samples, samples_pmid



if not (os.path.isfile('input/pos_samples') \
        and os.path.isfile('input/neg_samples') ):
  print 'Replacing ontologies...'
  pos_samples, pos_samples_pmid = replaceOntology(pmid2sct, pos_corpus[0:170665], g2c, d2c)
  neg_samples, neg_samples_pmid = replaceOntology(pmid2sct, neg_corpus[0:170665], g2c, d2c)
  import json
  with open('input/pos_samples', 'w') as f:
    json.dump(pos_samples,f)
  with open('input/neg_samples', 'w') as f:
    json.dump(neg_samples,f)
  with open('input/pos_samples_pmid', 'w') as f:
    json.dump(pos_samples_pmid,f)
  with open('input/neg_samples_pmid', 'w') as f:
    json.dump(neg_samples_pmid,f)
    