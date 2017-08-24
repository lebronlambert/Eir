__author__ = 'Xiang Liu'

import sys

sys.path.append('../')
from for_model import *
from for_data import *
from updateState import *

import warnings
warnings.filterwarnings("ignore")

# newstate, entity = updatestate(run_title, run_journal, run_index_truth, run_author, abstract_mode)
# newstate = updatestate(run_abstract, None, None,None, title_mode)

def updatestate(text,append1,append2,append3,mode):
    state=[0 for i in range(56)]
    entity,confidence,context1,context2=run_return_state(text)
    # print entity
    # print confidence
    # print  context1
    # print context2
    if mode==True:
        if entity[0]==u'GENE':
            state[0]=1
        if entity[1]==u'TRAIT':
            state[1]=1
        state[2:4]=confidence
        state[4:10]=context1[0]
        state[10:16]=context1[1]
        state[16:22]=context2[0]
        state[22:28]=context2[1]
        state[52:55] = append3
        state[55] = append1
    else:
        # if entity[0]==u'GENE':
        #     state[0]=1
        # if entity[1]==u'TRAIT':
        #     state[1]=1
        # state[2:4] = confidence
        state[28:34] = context1[0]
        state[34:40] = context1[1]
        state[40:46] = context2[0]
        state[46:52] = context2[1]
        state[52:55]=append3
        state[55]=append1
    print state
    return entity,state




def judge_ground_truth(CORRECT , PRED,GOLD,CORRECT2 , PRED2,GOLD2,CORRECT3 , PRED3,GOLD3,CORRECT4 , PRED4,GOLD4,ground_truth,action,abstract_mode,run_index_truth,entity):
    #print ground_truth,action,abstract_mode,run_index_truth
    if abstract_mode==True:
        if action==1 and ground_truth=='1':
            CORRECT+=1
            if entity[0]==u'GENE':
                CORRECT3+=1
            if entity[1]==u'TRAIT':
                CORRECT4+=1
            PRED3 += 1
            PRED4 += 1
            GOLD3+=1
            GOLD4+=1

        if action==1:
            PRED+=1

        if ground_truth=='1':
            GOLD+=1

    else:
        if action == 1 and ground_truth == '1':
            CORRECT2+= 1
        if action == 1:
            PRED2 += 1
        if ground_truth == '1':
            GOLD2 += 1
    return CORRECT, PRED, GOLD, CORRECT2, PRED2, GOLD2, CORRECT3, PRED3, GOLD3, CORRECT4, PRED4, GOLD4


def calculate_reward(action,ground_truth):
    ##print run_groundtruth, action, abstract_mode, run_index_truth, entity
    reward=0
    if action==2 and ground_truth=='1':
        print "reward1"
        reward=-5.66  #0.1 output
    elif action==1 and ground_truth=='1':
        print "reward2"
        reward=5.66
    elif action==2 and ground_truth=='0':
        print "reward3"
        reward=1.
    elif action==1 and ground_truth=='0':
        print "reward4"
        reward=-0.9#7004  -0.1  #7003 -1 # 7005  -10

    return reward

def main(args):
    #WORD_LIMIT = 500
    print args.outFile
    outFile = open( args.outFile, 'w', 0)
    outFile.write(str(args) + "\n")

    DEBUG=False
    TRAIN_abstract, TRAIN_author, TRAIN_groundtruth, TRAIN_journal, TRAIN_title, TRAIN_index,TEST_abstract, TEST_author, TEST_groundtruth, TEST_journal, TEST_title,TEST_index=FOR_data()
    print len(TRAIN_abstract)
    print len(TEST_abstract)

    abstract=TRAIN_abstract
    author=TRAIN_author
    groundtruth=TRAIN_groundtruth
    journal=TRAIN_journal
    title=TRAIN_title
    index_truth=TRAIN_index

    # abstract=TEST_abstract
    # author=TEST_author
    # groundtruth=TEST_groundtruth
    # journal=TEST_journal
    # title=TEST_title
    # index_truth=TEST_index

    # server setup
    port = args.port
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    evaluate_mode=False
    abstract_mode=True
    print "Started server on port", port

    # for analysis
    stepCnt = 0

    #initialize
    articleNum=0
    CORRECT , PRED,GOLD=0.,0.,0.
    CORRECT2, PRED2, GOLD2 = 0., 0., 0.
    CORRECT3, PRED3, GOLD3=0.,0.,0.
    CORRECT4, PRED4, GOLD4=0.,0.,0.
    savedArticleNum=0
    number_for_one_trait,begin_number_for_one_trait=0,0
    run_index_truth = None
    run_author = None
    run_groundtruth = None
    run_journal = None
    run_title = None
    run_abstract =None
    entity=None
    newstate=[0 for i in range(56)]
    print len(newstate)
    print "~~~~~~~~~~~~"
    reward=0.
    terminal=''

    while True:
        #  Wait for next request from client

        message = socket.recv()
        #print message
        # print "Received request: ", message

        if message == "newGame":

            indx = articleNum % len(abstract)
            abstract_mode=True

            articleNum += 1

            number_for_one_trait=len(abstract[indx])
            begin_number_for_one_trait=0
            # print "indx:", indx, number_for_one_trait
            # if DEBUG:
            #     print "indx:", indx, number_for_one_trait

            run_index_truth = index_truth[indx][0]
            run_author=author[indx][0]
            run_groundtruth=groundtruth[indx][0]  #
            run_journal=journal[indx][0]
            run_title=title[indx][0]
            run_abstract=abstract[indx][0]
            # entity,newstate=updatestate(run_title,run_journal,run_index_truth,run_author,abstract_mode)
            #newstate = updatestate(run_abstract, None, None,None, title_mode)
            reward, terminal =  0, 'false'

        elif message == "evalStart":
            CORRECT, PRED, GOLD = 0, 0, 0
            CORRECT2, PRED2, GOLD2 = 0, 0, 0
            CORRECT3, PRED3, GOLD3=0,0,0
            CORRECT4, PRED4, GOLD4=0,0,0
            evaluate_mode = True
            stepCnt = 0
            savedArticleNum=articleNum
            abstract=TEST_abstract
            author=TEST_author
            groundtruth=TEST_groundtruth
            journal=TEST_journal
            title=TEST_title
            index_truth = TEST_index

            print "##### Evaluation Started ######"

        elif message == "evalEnd":
            print message
            print "------------\nEvaluation Stats: (Precision, Recall, F1):"
            # print PRED,GOLD,CORRECT
            # print PRED2, GOLD2, CORRECT2
            # print PRED3, GOLD3, CORRECT3
            # print PRED4, GOLD4, CORRECT4
            if PRED==0 or GOLD ==0:
                print "====0===="
            else:
                prec = float(CORRECT)/ PRED
                rec = float(CORRECT)/ GOLD
                f1 = (2 * prec * rec) / (prec + rec)
                print prec, rec, f1,"#",CORRECT,PRED,GOLD

            if PRED2 == 0 or GOLD2 == 0:
                print "====0===2"
            else:
                prec = float(CORRECT2) / PRED2
                rec =float( CORRECT2) / GOLD2
                f1 = (2 * prec * rec) / (prec + rec)
                print prec, rec, f1, "#", CORRECT2, PRED2, GOLD2

            if PRED3 == 0 or GOLD3 == 0:
                print "====0===3"
            else:

                prec =float( CORRECT3) / PRED3
                rec = float(CORRECT3) / GOLD3
                f1 = (2 * prec * rec) / (prec + rec)
                print prec, rec, f1, "#", CORRECT3, PRED3, GOLD3

            if PRED4 == 0 or GOLD4 == 0:
                print "====0===4"
            else:
                prec =float( CORRECT4 )/ PRED4
                rec =float( CORRECT4 )/ GOLD4
                f1 = (2 * prec * rec) / (prec + rec)
                print prec, rec, f1, "#", CORRECT4, PRED4, GOLD4

            print stepCnt,"stepCnt"


            print "StepCnt (total, average):", stepCnt, float(stepCnt) / len(abstract)

            outFile.write("------------\nEvaluation Stats: (Precision, Recall, F1):\n")
            outFile.write(' '.join([str(CORRECT),str(PRED),str(GOLD)]) + '\n')
            outFile.write(' '.join([str(CORRECT2),str(PRED2),str(GOLD2)]) + '\n')
            outFile.write(' '.join([str(CORRECT3),str(PRED3),str(GOLD3)]) + '\n')
            outFile.write(' '.join([str(CORRECT4),str(PRED4),str(GOLD4)]) + '\n')
            outFile.write("StepCnt (total, average): " + str(stepCnt) + ' ' + str(float(stepCnt) / len(abstract)) + '\n')

            abstract = TRAIN_abstract
            author = TRAIN_author
            groundtruth = TRAIN_groundtruth
            journal = TRAIN_journal
            title = TRAIN_title
            index_truth = TRAIN_index


            articleNum = savedArticleNum
            evaluate_mode = False



        else:
            action, query = [int(q) for q in message.split()]
            action=query


            # if action==1:


            reward = calculate_reward(action, run_groundtruth)

            if abstract_mode:
                entity,newstate=updatestate(run_abstract, run_journal, run_index_truth, run_author,abstract_mode)
            else:
                entity,newstate=updatestate(run_title, run_journal, run_index_truth, run_author, abstract_mode)

            if evaluate_mode:
                # if action==1:
                print run_groundtruth, action, abstract_mode, run_index_truth, entity
                print "~~~~~~~~~"
            else:
                # if action==1:
                print run_groundtruth, action, abstract_mode, run_index_truth, entity
                print "----------"

            if evaluate_mode == True:
                #print abstract_mode,"----"
                CORRECT, PRED, GOLD, CORRECT2, PRED2, GOLD2, CORRECT3, PRED3, GOLD3, CORRECT4, PRED4, GOLD4=judge_ground_truth(CORRECT, PRED, GOLD, CORRECT2, PRED2, GOLD2 ,CORRECT3, PRED3, GOLD3, CORRECT4, PRED4, GOLD4,run_groundtruth, action, abstract_mode,run_index_truth,entity)
                print CORRECT, PRED, GOLD, CORRECT2, PRED2, GOLD2, CORRECT3, PRED3, GOLD3, CORRECT4, PRED4, GOLD4

            if abstract_mode==False and action==1:
                abstract_mode=True
            else:
                abstract_mode=True
                begin_number_for_one_trait+=1
                if begin_number_for_one_trait==number_for_one_trait:
                    terminal='true'
                else:
                    run_index_truth = index_truth[indx][begin_number_for_one_trait]
                    run_author = author[indx][begin_number_for_one_trait]
                    run_groundtruth = groundtruth[indx][begin_number_for_one_trait]  #
                    run_journal = journal[indx][begin_number_for_one_trait]
                    run_title = title[indx][begin_number_for_one_trait]
                    run_abstract = abstract[indx][begin_number_for_one_trait]


            # print "number:", begin_number_for_one_trait,number_for_one_trait






            # terminal = 'true' if terminal else 'false'

            # if evaluate_mode==False:
            #     print abstract_mode

        if message != "evalStart" and message != "evalEnd":
            if evaluate_mode==True:
                print message
                print "stepCnt:",stepCnt
                stepCnt += 1


            outMsg = 'state, reward, terminal = ' + str(newstate) + ',' + str(reward) + ',' + terminal
            socket.send(outMsg.replace('[', '{').replace(']', '}'))
            #print "last:",terminal,reward
        else:
            socket.send("done")



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--port",
                           type=int,
                           default=7010,
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
                           default='./output/out7010',
                           help="Output File")
    #
    # argparser.add_argument("--evalOutFile",
    #                        default="",
    #                        type=str,
    #                        help="Output File for predictions")
    #
    # argparser.add_argument("--modelFile",
    #                        type=str,
    #                        default='trained_model2.p',
    #                        help="Model File")
    #
    # argparser.add_argument("--shooterLenientEval",
    #                        type=bool,
    #                        default=True,  # here
    #                        help="Evaluate shooter leniently by counting any match as right")
    #
    # argparser.add_argument("--shooterLastName",
    #                        type=bool,
    #                        default=False,
    #                        help="Evaluate shooter using only last name")
    #
    # argparser.add_argument("--oracle",
    #                        type=bool,
    #                        default=False,
    #                        help="Evaluate using oracle")
    #
    # argparser.add_argument("--ignoreDuplicates",
    #                        type=bool,
    #                        default=False,
    #                        help="Ignore duplicate articles in downloaded ones.")
    #
    # argparser.add_argument("--baselineEval",
    #                        type=bool,
    #                        default=False,
    #                        help="Evaluate baseline performance")
    #
    # argparser.add_argument("--classifierEval",
    #                        type=bool,
    #                        default=False,
    #                        help="Evaluate performance using a simple maxent classifier")
    #
    # argparser.add_argument("--thresholdEval",
    #                        type=bool,
    #                        default=False,
    #                        help="Use tf-idf similarity threshold to select articles to extract from")
    #
    # argparser.add_argument("--threshold",
    #                        type=float,
    #                        default=0.8,
    #                        help="threshold value for Aggregation baseline above")
    #
    # argparser.add_argument("--confEval",
    #                        type=bool,
    #                        default=False,
    #                        help="Evaluate with best conf ")
    #
    # argparser.add_argument("--rlbasicEval",
    #                        type=bool,
    #                        default=False,
    #                        help="Evaluate with RL agent that takes only reconciliation decisions.")
    #
    # argparser.add_argument("--rlqueryEval",
    #                        type=bool,
    #                        default=False,
    #                        help="Evaluate with RL agent that takes only query decisions.")
    #
    # argparser.add_argument("--shuffleArticles",
    #                        type=bool,
    #                        default=False,
    #                        help="Shuffle the order of new articles presented to agent")
    #
    # argparser.add_argument("--entity",
    #                        type=int,
    #                        default=4,
    #                        help="Entity num. 4 means all.")
    #
    # argparser.add_argument("--aggregate",
    #                        type=str,
    #                        default='always',
    #                        help="Options: always, conf, majority")
    #
    # argparser.add_argument("--delayedReward",
    #                        type=str,
    #                        default='False',
    #                        help="delay reward to end")
    #
    #
    # argparser.add_argument("--numEntityLists",
    #                        type=int,
    #                        default=1,
    #                        help="number of different query lists to consider")
    #
    # argparser.add_argument("--contextType",
    #                        type=int,
    #                        default=2,
    #                        help="Type of context to consider (1 = counts, 2 = tfidf, 0 = none)")
    #
    # argparser.add_argument("--saveEntities",
    #                        type=bool,
    #                        default=False,
    #                        help="save extracted entities to file")


    args = argparser.parse_args()
    print args
    main(args)



