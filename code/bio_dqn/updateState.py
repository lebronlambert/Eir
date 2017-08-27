__author__ = 'Xiang Liu'
import numpy as np

import sys
sys.path.append('../')
import copy
import sys, json, pdb, pickle, operator, collections
from itertools import izip
import inflect
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import itertools
# from for_data import *

trained_model='./trained_model_4000.p'
if type(trained_model) == str:
    clf,  word_vocab, other_features = pickle.load(open(trained_model, "rb"))
else:
    clf,  word_vocab, other_features = trained_model

int2tags = ['TAG','trait', 'gene']
tags2int = {'TAG': 0, \
            'trait': 1, \
             'gene': 2}#, \
            # 'woundedNum': 3,\
            # 'city': 4}


inflect_engine = inflect.engine()


WORD_LIMIT = 1000
tfidf_vectorizer = TfidfVectorizer()


CONTEXT_LENGTH = 3



def computeContext(ENTITIES, CONTEXT, ARTICLES,  vectorizer, context=3):
    # print "Computing context..."
    vocab = vectorizer.vocabulary_
    article = [w for w in ARTICLES[0]]
    ii=0
    for entity in ENTITIES:
        ii += 1
        if ii <= 2:
            for i, word in enumerate(article):


                vec = []
                phrase = []
                if type(word) == int or word.isdigit() and word < 100:
                    word = int(word)
                    word = inflect_engine.number_to_words(word)
                if word in entity:
                    for j in range(1, context + 1):
                        if i - j >= 0:
                            phrase.append(article[i - j])
                        else:
                            phrase.append('XYZUNK')  # random unseen phrase
                    for j in range(1, context + 1):
                        if i + j < len(article):
                            phrase.append(article[i + j])
                        else:
                            phrase.append('XYZUNK')  # random unseen phrase
                    break

            mat = vectorizer.transform([' '.join(phrase)]).toarray()
            for w in phrase:
                feat_indx = vocab.get(w)
                if feat_indx:
                    vec.append(float(mat[0, feat_indx]))
                else:
                    vec.append(0.)

            if len(vec) == 0:
                vec = [0. for q in range(2 * context)]
            CONTEXT.append(vec)

    # print "done."
    return CONTEXT


def extractEntitiesWithConfidences(clf,word_vocab,article):
    joined_article = ' '.join(article)

    pred, conf_scores, conf_cnts = predictWithConfidences(clf,word_vocab, joined_article, False)

    for i in range(len(conf_scores)):
        if conf_cnts[i] > 0:
            conf_scores[i] /= conf_cnts[i]

    result = pred.split(' ### '), conf_scores
    split =  [" ### " in r for r in result]
    assert sum(split) == 0
    return result


def predictWithConfidences(clf,word_vocab, sentence, viterbi):
    sentence = sentence.replace("_"," ")

    words = re.findall(r"[\w']+|[.,!?;]", sentence)
    cleanedSentence = []

    i = 0
    while i < len(words):
        token = sentence[i]
        end_token_range = i
        for j in range(i+1,len(words)):
            new_token = words[j]
            if new_token == token:
                end_token_range = j
            else:
                cleanedSentence.append(words[i])
                break
        i = end_token_range + 1

    words = cleanedSentence
    # other_features=helper.other_features
    # trained_model = '../trained_model2.p'
    # if type(trained_model) == str:
    #     clf, previous_n,next_n, word_vocab,other_features = pickle.load( open( trained_model, "rb" ) )
    # else:
    #     clf, previous_n,next_n, word_vocab,other_features = trained_model
    global  other_features
    previous_n=0
    next_n=4
    y, confidences = predict_tags_n(viterbi, previous_n,next_n, clf, words, word_vocab,other_features)
    tags = []
    for i in range(len(y)):
        tags.append(int(y[i]))

    pred, conf_scores, conf_cnts = predict_mode(words, tags, confidences)

    return pred, conf_scores, conf_cnts


def predict_mode(sentence, tags, confidences, crf=False):
    output_entities = {}
    entity_confidences = [0, 0]
    entity_cnts = [0, 0]

    for tag in int2tags:
        # print tag
        output_entities[tag] = []

    for j in range(len(sentence)):
        ind = ""
        # if crf:
        #     ind = tags[j]
        # else:
        # print tags[j]
        ind = int2tags[tags[j]]
        # print ind
        output_entities[ind].append((sentence[j], confidences[j]))

    output_pred_line = ""

    mode, conf = get_mode(output_entities["trait"])

    output_pred_line += mode
    entity_confidences[tags2int['trait'] - 1] += conf
    entity_cnts[tags2int['trait'] - 1] += 1

    output_pred_line += " ### "

    mode, conf = get_mode(output_entities["gene"])

    output_pred_line += mode
    entity_confidences[tags2int['gene'] - 1] += conf
    entity_cnts[tags2int['gene'] - 1] += 1


    # for tag in int2tags:
    #
    #     if  tag not in ["TAG", "trait"]:
    #
    #
    #         mode, conf = get_mode(output_entities[tag])
    #         if mode == "":
    #             output_pred_line += "zero"
    #             entity_confidences[tags2int[tag] - 1] += 0
    #             entity_cnts[tags2int[tag] - 1] += 1
    #         else:
    #             output_pred_line += mode
    #             entity_confidences[tags2int[tag] - 1] += conf
    #             entity_cnts[tags2int[tag] - 1] += 1

    #assert not (output_pred_line.split(" ### ")[0].strip() == "" and len(output_entities["gene"]) > 0)

    return output_pred_line, entity_confidences, entity_cnts


def get_mode(l):
    p = inflect.engine()
    counts = collections.defaultdict(lambda:0)
    Z = collections.defaultdict(lambda:0)
    curr_max = 0
    arg_max = ""
    for element, conf in l:
        try:
            normalised = p.number_to_words(int(element))
        except Exception, e:
            normalised = element


        counts[normalised] += conf
        Z[normalised] += 1

    for element in counts:
        if counts[element] > curr_max and element != "" and element != "zero":
            curr_max = counts[element]
            arg_max = element
    return arg_max, (counts[arg_max]/Z[arg_max] if Z[arg_max] > 0 else counts[arg_max])



def predict_tags_n(viterbi, previous_n,next_n, clf, sentence, word_vocab,other_features,first_n = 10):

    num_features = len(word_vocab) + len(other_features)

    total_features = (previous_n + next_n +1)*num_features + len(word_vocab) + first_n
    # print total_features
    dataX = np.zeros((len(sentence),total_features))
    dataY = np.zeros(len(sentence))
    dataYconfidences = [None for i in range(len(sentence))]
    #other_words_lower = set([s.lower() for s in sentence[0]])
    other_words_lower = set([s for s in sentence[0]])

    for i in range(len(sentence)):
        word = sentence[i]
        #word_lower = word.lower()
        word_lower = word
        if word_lower in word_vocab:
        #if word in word_vocab:
            dataX[i,word_vocab[word_lower]] = 1
            for j in range(next_n):
                if i-j-1 >= 0:
                    dataX[i-j-1,(j+1)*num_features+word_vocab[word_lower]] = 1

        for (index, feature_func) in enumerate(other_features):  #here!!
            if feature_func(word):
                dataX[i,len(word_vocab)+index] = 1
                for j in range(next_n):
                    if i - j - 1 >= 0:
                        dataX[i-j-1,(j+1)*num_features+len(word_vocab)+index] = 1

        for other_word_lower in other_words_lower:
            if other_word_lower != word_lower and other_word_lower in word_vocab:
                dataX[i,(next_n+1)*num_features + word_vocab[other_word_lower]] = 1
        if i < first_n:
            dataX[i,( next_n + 1)*num_features + len(word_vocab) + i ] = 1

    for i in range(len(sentence)):
        dataYconfidences[i] = clf.predict_proba(dataX[i,:])
        dataY[i] = np.argmax(dataYconfidences[i])
        dataYconfidences[i] = dataYconfidences[i][0][int(dataY[i])]
        # print dataY[i]

    return dataY, dataYconfidences


def run_return_state(text): #,number
    if len(text[0])==0:
        return ['',''],[0,0],[[0,0,0,0,0,0],[0,0,0,0,0,0]],[[0,0,0,0,0,0],[0,0,0,0,0,0]]  #waiting


    articles_= [' '.join(tokens) for tokens, tags in [text]]

    vectorizer1 = CountVectorizer(min_df=1)
    vectorizer2 = TfidfVectorizer(min_df=1)
    vectorizer1.fit(articles_)
    vectorizer2.fit(articles_)

    entities, confidences = extractEntitiesWithConfidences(clf,word_vocab,text[0])
    # print entities
    # print confidences

    CONTEXT1 = []
    CONTEXT2 = []


    computeContext(entities, CONTEXT1, text, vectorizer1, CONTEXT_LENGTH)
    computeContext(entities, CONTEXT2, text, vectorizer2, CONTEXT_LENGTH)
    # print CONTEXT1
    # print CONTEXT2
    #print entities,confidences,CONTEXT1,CONTEXT2
    return entities,confidences,CONTEXT1,CONTEXT2


if __name__ == '__main__':
    for i in range(20):
        number=i
        run_return_state(None,number)
