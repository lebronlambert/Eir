__author__ = 'Xiang Liu'
import numpy as np

import sys
sys.path.append('../')
import copy
import sys, json, pdb, pickle, operator, collections
from itertools import izip
import inflect
import helper
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from server import loadFile
import re
import itertools

NUM_ENTITIES=4
def loadFile(filename):
    articles, titles, identifiers, downloaded_articles = [], [], [], []

    # load data and process identifiers
    with open(filename, "rb") as inFile:
        while True:
            try:
                a, b, c, d = pickle.load(inFile)
                articles.append(a)
                titles.append(b)
                identifiers.append(c)
                downloaded_articles.append(d)
            except:
                break

    identifiers_tmp = []
    for e in identifiers:
        for i in range(NUM_ENTITIES):
            if type(e[i]) == int or e[i].isdigit():
                e[i] = int(e[i])
                e[i] = inflect_engine.number_to_words(e[i])

        identifiers_tmp.append(e)
    identifiers = identifiers_tmp

    return articles, titles, identifiers, downloaded_articles

# import constants
#
# int2tags = ['TAG'] + constants.int2tags
# tags2int = constants.tags2int
# print int2tags
# print tags2int
int2tags = ['shooterName', 'killedNum', 'woundedNum', 'city']
int2tags = ['TAG'] + int2tags
tags2int = {'TAG': 0, \
            'shooterName': 1, \
             'killedNum': 2, \
            'woundedNum': 3,\
            'city': 4}
print int2tags
print tags2int

#  int2tags = ['TAG'] + ['shooterName', 'killedNum']
# tags2int = {'TAG': 0,
#             'shooterName': 1,
#             'killedNum': 2, }

inflect_engine = inflect.engine()


WORD_LIMIT = 1000
tfidf_vectorizer = TfidfVectorizer()


CONTEXT_LENGTH = 3

#must train one model hhhhhhhhhhhhhhhhhhhhhhhhhhhhh
# and must for a useful dataset


def computeContext(ENTITIES, CONTEXT, ARTICLES,  vectorizer, context=3):
    print "Computing context..."
    vocab = vectorizer.vocabulary_
    article = [w.lower() for w in ARTICLES[0]]
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

    print "done."
    return CONTEXT


def extractEntitiesWithConfidences(clf,word_vocab,article):
    joined_article = ' '.join(article)

    pred, conf_scores, conf_cnts = predictWithConfidences(clf,word_vocab, joined_article, False, helper.cities)

    for i in range(len(conf_scores)):
        if conf_cnts[i] > 0:
            conf_scores[i] /= conf_cnts[i]

    result = pred.split(' ### '), conf_scores
    split =  [" ### " in r for r in result]
    assert sum(split) == 0
    return result


def predictWithConfidences(clf,word_vocab, sentence, viterbi, cities):
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
    trained_model = '../trained_model2.p'
    if type(trained_model) == str:
        clf, previous_n,next_n, word_vocab,other_features = pickle.load( open( trained_model, "rb" ) )
    else:
        clf, previous_n,next_n, word_vocab,other_features = trained_model

    previous_n=0
    next_n=4
    y, confidences = predict_tags_n(viterbi, previous_n,next_n, clf, words, word_vocab,other_features)
    tags = []
    for i in range(len(y)):
        tags.append(int(y[i]))

    pred, conf_scores, conf_cnts = predict_mode(words, tags, confidences, cities)

    return pred, conf_scores, conf_cnts


def predict_mode(sentence, tags, confidences, cities, crf=False):
    output_entities = {}
    entity_confidences = [0, 0,0,0]
    entity_cnts = [0, 0,0,0]

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

    mode, conf = get_mode(output_entities["shooterName"])

    output_pred_line += mode
    entity_confidences[tags2int['shooterName'] - 1] += conf
    entity_cnts[tags2int['shooterName'] - 1] += 1

    for tag in int2tags:
        if tag == "city":
            output_pred_line += " ### "
            possible_city_combos = []
            # pdb.set_trace()
            for permutation in itertools.permutations(output_entities[tag], 2):
                if permutation[0][0] in cities:
                    if "" in cities[permutation[0][0]]:
                        possible_city_combos.append((permutation[0][0], permutation[0][1]))
                    if permutation[1][0] in cities[permutation[0][0]]:
                        possible_city_combos.append((permutation[0][0] + " " + permutation[1][0], \
                                                     max(permutation[0][1], permutation[1][1])))
            mode, conf = get_mode(possible_city_combos)

            # try cities automatically
            if mode == "":
                possible_cities = []
                for i in range(len(sentence)):
                    word1 = sentence[i]
                    if word1 in cities:
                        if "" in cities[word1]:
                            possible_cities.append((word1, 0.))
                        if i + 1 < len(sentence):
                            word2 = sentence[i + 1]
                            if word2 in cities[word1]:
                                possible_cities.append((word1 + " " + word2, 0.))

                # print possible_cities
                # print get_mode(possible_cities)
                mode, conf = get_mode(possible_cities)

            output_pred_line += mode
            entity_confidences[tags2int['city'] - 1] += conf
            entity_cnts[tags2int['city'] - 1] += 1

        elif tag not in ["TAG", "shooterName"]:
            output_pred_line += " ### "

            mode, conf = get_mode(output_entities[tag])
            if mode == "":
                output_pred_line += "zero"
                entity_confidences[tags2int[tag] - 1] += 0
                entity_cnts[tags2int[tag] - 1] += 1
            else:
                output_pred_line += mode
                entity_confidences[tags2int[tag] - 1] += conf
                entity_cnts[tags2int[tag] - 1] += 1

    assert not (output_pred_line.split(" ### ")[0].strip() == "" and len(output_entities["shooterName"]) > 0)

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
            normalised = element.lower()


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
    print total_features
    dataX = np.zeros((len(sentence),total_features))
    dataY = np.zeros(len(sentence))
    dataYconfidences = [None for i in range(len(sentence))]
    other_words_lower = set([s.lower() for s in sentence[0]])

    for i in range(len(sentence)):
        word = sentence[i]
        word_lower = word.lower()
        if word_lower in word_vocab:
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


def run(all_number):

    trained_model = '../trained_model2.p'
    if type(trained_model) == str:
        clf, previous_n,next_n, word_vocab,other_features = pickle.load( open( trained_model, "rb" ) )
    else:
        clf, previous_n,next_n, word_vocab,other_features = trained_model
    fileName = '../dloads/Shooter/dev.extra'
    numLists = 1
    sysargv4 = '../consolidated/vec_dev_1.5.p'
    listNum = 0

    articles, titles, identifiers, downloaded_articles = loadFile(fileName + '.' + str(listNum))
    #print articles[all_number]
    articles_ = [' '.join(tokens) for tokens, tags in [articles[all_number]]]
    vectorizer1 = CountVectorizer(min_df=1)
    vectorizer2 = TfidfVectorizer(min_df=1)
    vectorizer1.fit(articles_)
    vectorizer2.fit(articles_)
    articles, titles, identifiers, downloaded_articles = loadFile(fileName + '.' + str(0))

    ARTICLES = articles[all_number]
    TITLES = titles[all_number]
    IDENTIFIERS = identifiers[all_number]
    DOWNLOADED_ARTICLES = [[] for q in range(1)]
    DOWNLOADED_ARTICLES[0].append(downloaded_articles[all_number])

    temp=ARTICLES
    # temp=TITLES.split(" ")

    entities, confidences = extractEntitiesWithConfidences(clf,word_vocab,temp[0])
    print entities
    print confidences

    CONTEXT1 = []
    CONTEXT2 = []

    computeContext(entities, CONTEXT1, temp, vectorizer1, CONTEXT_LENGTH)
    computeContext(entities, CONTEXT2, temp, vectorizer2, CONTEXT_LENGTH)
    print CONTEXT1
    print CONTEXT2


#
for i in range(2):
    run(i)
    print "-----------------"

#
# def main(training_file,trained_model,previous_n,next_n, c, prune, test_file):
#     helper.load_constants()
#     train_data, identifier = load_data(training_file)
#     print len(train_data)
#     print len(identifier)
#     print train_data[0]
#     print len(train_data[0][0])
#     print train_data[0][0]
#     print train_data[0][1]
#     print len(train_data[0])
#     print identifier[0]
#     print
#     print
#
#     print len(train_data)
#     print len(identifier)
#     print train_data[1]
#     print len(train_data[1][0])
#     print train_data[1][1]
#     print len(train_data[1])
#     print identifier[1]
#     test_data, test_ident = load_data(test_file)
#     print len(test_data)
#     print len(test_ident)
#     print test_data[0]
#     print len(test_data[0])
#     print test_data[0][0:100]
#     print test_ident[0]
#     ## extract features
#     tic = time.clock()
#     print "get word_vocab"
#     # print train_data
#     num_words, word_vocab = get_word_vocab(train_data, prune)
#     print num_words
#     print word_vocab
#     print "feature extract for train"
#     trainX, trainY = get_feature_matrix_n(previous_n,next_n,train_data, num_words, word_vocab, helper.other_features)
#     # print trainX
#     # print trainY
#     print 'feature extract for test'
#     testX, testY   = get_feature_matrix_n(previous_n, next_n, test_data, num_words, word_vocab, helper.other_features)
#     print time.clock()-tic
#
#     ## train LR
#     print("training")
#     tic = time.clock()
#     clf = LogisticRegression(C=c, multi_class='multinomial', solver='lbfgs')
#     clf.fit(trainX,trainY)
#     print trainX.shape
#     print trainY.shape
#     print time.clock()-tic
#
#     print "predicting"
#     predictY = clf.predict(testX)
#     assert len(predictY) == len(testY)
#
#     print "evaluating"
#
#
#     return [clf, previous_n,next_n, word_vocab,helper.other_features]

