__author__ = 'Xiang Liu'


import sys

sys.path.append('../')
import pickle

import zmq, time
import numpy as np
import copy
import  json, pdb, pickle, operator, collections

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from random import shuffle
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LogisticRegression
import scipy.sparse
import helper

def get_feature_matrix_n(previous_n,next_n,data, num_words, word_vocab, other_features,first_n=10):
    print "begin"
    num_features = len(word_vocab) + len(other_features)
    total_features = (previous_n+next_n+1)*num_features + len(word_vocab) + previous_n  + first_n
    # print "~~~~~~~~~~~~~~~~~~~"
    # print total_features
    #print num_words, num_features, total_features
    dataY = np.zeros(num_words)
    dataX = scipy.sparse.lil_matrix((num_words, total_features))
    print "action"
    curr_word = 0
    print len(data)
    i_count=0
    for sentence in data:
        i_count+=1
        print i_count,sentence[0][0:5],sentence[1][0:5]
        #print sentence
        #other_words_lower = set([s.lower for s in sentence[0]])
        # if len(sentence)==0:
        #     pass
        # else:
        other_words_lower = set([s for s in sentence[0]])
        for i in range(len(sentence[0])):
            word = sentence[0][i]
            #word_lower = word.lower()
            word_lower=word
            if word_lower in word_vocab:
                dataX[curr_word,word_vocab[word_lower]] = 1
                for j in range(previous_n):
                    if i+j+1<len(sentence[0]):
                        dataX[curr_word+j+1,(j+1)*num_features+word_vocab[word_lower]] = 1
                for j in range(next_n):
                    if i-j-1 >= 0:
                        dataX[curr_word-j-1,(previous_n+j+1)*num_features+word_vocab[word_lower]] = 1
            for (index, feature_func) in enumerate(other_features):
                if feature_func(word):
                    dataX[curr_word,len(word_vocab)+index] = 1
                    for j in range(previous_n):
                        if i + j + 1 < len(sentence[0]):
                            dataX[curr_word+j+1,(j+1)*num_features+len(word_vocab)+index] = 1
                    for j in range(next_n):
                        if i - j - 1 >= 0:
                            dataX[curr_word-j-1,(previous_n+j+1)*num_features+len(word_vocab)+index] = 1
            for other_word_lower in other_words_lower:
                if other_word_lower != word_lower and other_word_lower in word_vocab:
                    dataX[curr_word,(previous_n+next_n+1)*num_features + word_vocab[other_word_lower]] = 1

            if i < first_n:
                dataX[curr_word,(previous_n+next_n+1)*num_features + len(word_vocab) +i] = 1
            assert len(sentence[0]) == len(sentence[1])
            dataY[curr_word] = sentence[1][i]
            curr_word += 1
    return dataX, dataY

def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a==item]

def return_vector_for_model():
    f_ground_truth = open('./data/ground_truth4000','r')
    data_ground_truth = pickle.load(f_ground_truth)

    f_data_merge = open('./data/ground_truth4000_output_cooked','r')
    data_merge = pickle.load(f_data_merge)
    # f_ground_truth = open('./DeepRL-data1000/ground_truth1000_cleaned','r')
    # data_ground_truth = pickle.load(f_ground_truth)
    # f_data_merge = open('./DeepRL-data1000/preprocessed_articles','r')
    # data_merge = pickle.load(f_data_merge)
    print len(data_merge)
    data_all_for_train=[]
    data_all_for_test=[]
    sum_temp = 1501
    sum=0
    for i in data_merge:
        #gene,trait=i
        # print gene,trait
        sum += 1
        if sum <= 1500:
            data_all_for_train.append([])
            for s in range(len(data_merge[i])):

                if s == 0:
                    print sum
                    temp=data_merge[i][s]['abstract'].split()
                    # print sum
                    data_all_for_train[sum - 1].append(temp)
                    if 'GENE' in temp:
                        temp_index1=find_all_index(temp,'GENE') #temp.index(i[0])
                    else:
                        temp_index1=-1
                    if 'TRAIT' in temp:
                        temp_index2 = find_all_index(temp, 'TRAIT')
                    else:
                        temp_index2=-1
                    data_all_for_train[sum - 1].append([0 for i in range(len(temp))])
                    if temp_index1!=-1:
                        for i_index in temp_index1:
                            data_all_for_train[sum - 1][1][i_index]=1
                    if temp_index2 != -1:
                        for i_index in temp_index2:
                            data_all_for_train[sum - 1][1][i_index] = 2

        elif sum<=2150:
            data_all_for_test.append([])
            for s in range(len(data_merge[i])):
                if s == 0:
                    print sum-1500
                    # print data_merge[i][s]['abstract']
                    temp = data_merge[i][s]['abstract'].split()
                    data_all_for_test[sum - sum_temp].append(temp)
                    # try:
                    #     temp_index1= temp.index(i[0])
                    #     temp_index2= temp.index(i[1])
                    #     print temp_index1
                    #     print temp_index2
                    # except:
                    #     print i[0] in temp
                    #     print i[1] in temp
                    #     temp_index1=-1
                    #     temp_index2=-1
                    if 'GENE'in temp:
                        # print sum,":1"
                        # print temp.index(i[0])
                        # print temp
                        temp_index1 = find_all_index(temp, 'GENE')  # temp.index(i[0])
                    else:
                        temp_index1 = -1
                    if 'TRAIT' in temp:
                        # print sum,":2"
                        # print temp.index(i[1])
                        # temp_index2=temp.index(i[1])
                        temp_index2 = find_all_index(temp, 'TRAIT')

                    else:
                        temp_index2 = -1
                    # print "~"
                    data_all_for_test[sum - sum_temp].append([0 for i in range(len(temp))])
                    if temp_index1 != -1:
                        for i_index in temp_index1:
                            # print i_index
                            data_all_for_test[sum - sum_temp][1][i_index] = 1
                    if temp_index2 != -1:
                        for i_index in temp_index2:
                            data_all_for_test[sum - sum_temp][1][i_index] = 2
                    #print data_all_for_test[sum - 701][1]
    num_words, word_vocab=get_word_vocab(data_all_for_train,5)
    # print len(data_all_for_train)
    # print len(data_all_for_test)
    # print data_all_for_train[0]
    # print len(data_all_for_train[0][0])
    # print data_all_for_train[0][0]
    # print data_all_for_train[0][1]
    # print len(data_all_for_train[0])
    #
    # print len(data_all_for_train)
    # print len(data_all_for_test)
    # print data_all_for_train[1]
    # print len(data_all_for_train[1][0])
    # print data_all_for_train[1][0]
    # print data_all_for_train[1][1]
    # print len(data_all_for_train[1])



    tic = time.clock()
    print "feature extract for train"
    previous_n=0
    next_n=4
    helper.load_constants()
    trainX, trainY = get_feature_matrix_n(previous_n, next_n, data_all_for_train, num_words, word_vocab, helper.other_features)
    print 'feature extract for test'
    testX, testY = get_feature_matrix_n(previous_n, next_n, data_all_for_test, num_words, word_vocab, helper.other_features)
    print time.clock() - tic
    print("training")
    # print trainX,trainY
    # print testX,testY
    # print trainX.shape
    # print trainY.shape
    # print testX.shape
    # print testY.shape
    tic = time.clock()
    clf = LogisticRegression(C=10, multi_class='multinomial', solver='lbfgs')
    clf.fit(trainX, trainY)
    print time.clock() - tic
    print "predicting"
    predictY = clf.predict(testX)
    assert len(predictY) == len(testY)
    pickle.dump([clf, word_vocab, helper.other_features], open('./trained_model_4000.p', "wb"))
    return clf,word_vocab,helper.other_features






####here for train model
def get_word_vocab(data, prune):
    num_words = 0
    word_vocab = {}
    for sentence in data:
        words_in_sentence = set()
        if len(sentence)==0:
            pass
        else:
            for word in sentence[0]:
                #word_lower=word.lower()
                word_lower=word
                if word_lower in words_in_sentence:
                    continue
                if word_lower not in word_vocab:
                    word_vocab[word_lower] = 1
                else:
                    word_vocab[word_lower] += 1
                words_in_sentence.add(word_lower)
            num_words += len(sentence[0])
    feature_list = []
    prune_features(word_vocab,feature_list, prune)
    return num_words,word_vocab


def prune_features(feature_vocab, featureList, prune):
    for w in feature_vocab.keys():
        if feature_vocab[w] <= prune:
            feature_vocab.pop(w,None)
    index = 0
    for w in feature_vocab.keys():
        feature_vocab[w] = index
        featureList.append(w)
        index += 1

# def return_model(train_data,prune):
#     num_words, word_vocab = get_word_vocab(train_data, prune)
#     return word_vocab

####end here for train model

if __name__ == '__main__':
    return_vector_for_model()