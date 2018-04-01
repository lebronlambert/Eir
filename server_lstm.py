__author__ = 'Xiang Liu'
import sys

from data_utils import load_dataset, load_dictionaries, get_minibatch
from model import Seq2RelNet
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import warnings
import argparse
import zmq, time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # FOR WHICH GPU YOU USE
warnings.filterwarnings("ignore")
STATE_NUMBER = 1000

# you may want me to help you prepare the dataset...
is_cuda = torch.cuda.is_available()
if is_cuda:
    print ('Using cuda...')
else:
    print ('NOT using cuda...')


def prepare_data():
    with open('input/concept2idx', 'r') as f:
        concept2idx = json.load(f)
    with open('input/typ2idx', 'r') as f:
        typ2idx = json.load(f)
    return concept2idx, typ2idx


def prepare_testdata():
    with open('input/test_set', 'r') as f:
        test_set = json.load(f)
    return test_set


def prepapre_traindata():
    with open('input/train_set', 'r') as f:
        train_set = json.load(f)
    return train_set


def prepare_traintest_data():
    with open('input/test_set', 'r') as f:
        test_set = json.load(f)
    num1 = len(test_set['samples'])
    num2 = int(num1 * 0.3)
    num1 = num1 - num2
    return test_set, num1, num2  # here if we dont use train_set dataset,we use the number to split the dataset


def prepare_model(concept2idx, typ2idx):
    global is_cuda
    model = Seq2RelNet(padding_idx=concept2idx['SELF_DEFINED_PAD'],
                       vocab_size_concept=len(concept2idx),
                       dim_emb_concept=512,
                       dim_typ=len(typ2idx),
                       dim_hidden=1000,
                       bidirectional=True,
                       num_layers=2,
                       dropout=0.,
                       is_cuda=is_cuda)
    if is_cuda:
        model.cuda()
    model.load_state_dict(torch.load('output_trial_5/model_param_epoch_2_index_96000'))
    return model


def prepare_data_batch(index_test, data_set, concept2idx, typ2idx):
    test_batch_size = 1  # 64
    max_length = 300
    batch_data = get_minibatch(data_set, concept2idx, typ2idx, test_batch_size, max_length, index_test)
    return batch_data


def updatestate(batch_data, model):
    global STATE_NUMBER
    global is_cuda
    if is_cuda:
        outputs = model(batch_data['batch_gene_disease'].cuda(),
                        batch_data['batch_typ'].cuda(),
                        batch_data['batch_concept'].cuda())
        ## here for future use
        #   print model.h_t #['concept_embedding', 'encoder', 'fc1']
        #   print "h_t"
        #   print model.src_h
        #   print model.src_h_t
        #   print model.src_c_t

        labels = batch_data['batch_label'].cuda()
    else:
        outputs = model(batch_data['batch_gene_disease'],
                        batch_data['batch_typ'],
                        batch_data['batch_concept'])
        labels = batch_data['batch_label']
    _, indices = torch.max(outputs, 1)
    temp_transform = (model.h_t.data).cpu().numpy()
    newstate = [0 for i in range(STATE_NUMBER)]
    for i in range(STATE_NUMBER):
        newstate[i] = temp_transform[0][i]
    # here we do not use the lable (we also can use it)
    return newstate


def judge_ground_truth(batch_data_label, action, CORRECT, PRED, GOLD):
    print batch_data_label
    if action == 1 and batch_data_label == 1:
        CORRECT += 1
        PRED += 1
        GOLD += 1
    elif action == 1 and batch_data_label == 0:
        PRED += 1
    elif action == 2 and batch_data_label == 1:
        GOLD += 1
    elif action == 2 and batch_data_label == 0:
        pass
    return CORRECT, PRED, GOLD


def calculate_reward(batch_data_label, action):
    print batch_data_label
    reward = 0.
    if action == 1 and batch_data_label == 1:
        reward = 1.
    elif action == 1 and batch_data_label == 0:
        reward = -1.
    elif action == 2 and batch_data_label == 1:
        reward = -1.
    elif action == 2 and batch_data_label == 0:
        reward = -1.
    return reward


def main(args):
    global STATE_NUMBER
    # WORD_LIMIT = 500

    # here output for debug
    outFile = open(args.outFile + str(args.port), 'w',
                   0)  ##+'_'+str(args.title_and_abstract_model)+'_'+str(args.stop_model)
    outFile.write(str(args) + "\n")

    # server setup
    port = args.port
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    evaluate_mode = False
    ##abstract_mode=False #future
    print "Started server on port", port

    ##SHUFFLED ALREADY
    # shuffledIndxs = [range(len(q)) for q in abstract]
    # for q in shuffledIndxs:
    #     shuffle(q)


    # initialize
    articleNum = 0
    CORRECT, PRED, GOLD = 0., 0., 0.
    savedArticleNum = 0
    savedArticleNum2 = 0
    newstate = [0 for i in range(STATE_NUMBER)]
    reward = 0.
    concept2idx, typ2idx = prepare_data()
    train_set = prepapre_traindata()
    test_set = prepare_testdata()
    model = prepare_model(concept2idx, typ2idx)
    savedallAriticleNum = len(train_set['samples'])
    savedallAriticleNum2 = len(test_set['samples'])
    allArticleNum = savedallAriticleNum
    terminal = 'false'
    print "ready"

    ## stop_model=args.stop_model #for future use
    ## title_and_abstract_model=args.title_and_abstract_model  #for future use
    ## if title_and_abstract_model!=1:
    ##     abstract_mode=True

    while True:

        #  Wait for next request from client
        message = socket.recv()

        if message == "newGame":
            indx = articleNum % allArticleNum
            articleNum += 1
            batch_data = prepare_data_batch(indx, train_set, concept2idx, typ2idx)
            reward, terminal = 0, 'false'
            newstate = updatestate(batch_data, model)

        elif message == "evalStart":
            CORRECT, PRED, GOLD = 0, 0, 0
            evaluate_mode = True
            savedArticleNum = articleNum
            articleNum = savedArticleNum2
            allArticleNum = savedallAriticleNum2
            print "##### Evaluation Started ######"

        elif message == "evalEnd":
            print message
            print "------------\nEvaluation Stats: (Precision, Recall, F1):"
            if PRED == 0 or GOLD == 0 or CORRECT == 0:
                print "====0===="
            else:
                prec = float(CORRECT) / PRED
                rec = float(CORRECT) / GOLD
                f1 = (2 * prec * rec) / (prec + rec)
                print prec, rec, f1, "#", CORRECT, PRED, GOLD

            outFile.write("------------\nEvaluation Stats: (Precision, Recall, F1):\n")
            outFile.write(' '.join([str(CORRECT), str(PRED), str(GOLD)]) + '\n')
            savedArticleNum2 = articleNum
            articleNum = savedArticleNum
            evaluate_mode = False
            allArticleNum = savedallAriticleNum

        else:
            print message
            action, query = [int(q) for q in message.split()]
            action = query  ##999 is useless
            # if stop_model==False:
            #     action=query
            batch_data_label = int((batch_data['batch_label']).cpu())
            reward = calculate_reward(batch_data_label, action)

            if evaluate_mode == True:
                CORRECT, PRED, GOLD = judge_ground_truth(batch_data_label, action, CORRECT, PRED, GOLD)

            indx = articleNum % allArticleNum
            articleNum += 1
            if evaluate_mode == True:
                batch_data = prepare_data_batch(indx, test_set, concept2idx, typ2idx)
            else:
                batch_data = prepare_data_batch(indx, train_set, concept2idx, typ2idx)
            newstate = updatestate(batch_data, model)

        if message != "evalStart" and message != "evalEnd":
            outMsg = 'state, reward, terminal = ' + str(newstate) + ',' + str(reward) + ',' + terminal
            # print outMsg
            socket.send(outMsg.replace('[', '{').replace(']', '}'))
        else:
            socket.send("done")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--port",
                           type=int,
                           default=7001,  # 7012 for test
                           help="port for server")
    # argparser.add_argument("--trainFile",
    #                        type=str,
    #                        help="training File")
    # argparser.add_argument("--testFile",
    #                        type=str,
    #                        default="",
    #                        help="Testing File")
    argparser.add_argument("--outFile",
                           type=str,
                           default='./output/out',
                           help="Output File")
    #
    # argparser.add_argument("--title_and_abstract_model",
    #                        default=3,   # 1 first title then abstract 2.only abstract 3 title and abtract
    #                        type=int,
    #                        help="Output File for predictions")
    #
    # argparser.add_argument("--stop_model",
    #                        type=bool,
    #                        default=False,
    #                        help="Model File")

    args = argparser.parse_args()
    print args
    main(args)


