__author__ = 'Xiang Liu'
import pickle

def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a==item]

def FOR_data():


    # f_ground_truth = open('./DeepRL-data1000/ground_truth1000_cleaned','r')
    # data_ground_truth = pickle.load(f_ground_truth)
    # f_data_merge = open('./DeepRL-data1000/preprocessed_articles','r')
    # data_merge = pickle.load(f_data_merge)
    f_ground_truth = open('./data/ground_truth4000', 'r')
    data_ground_truth = pickle.load(f_ground_truth)
    f_data_merge = open('./data/ground_truth4000_output_cooked', 'r')
    data_merge = pickle.load(f_data_merge)

    # f_other_merge = open('./DeepRL-data1000/ground_truth1000query_len','r')
    # other_data_merge = pickle.load(f_other_merge)

    TRAIN_title=[] # title
    TEST_title=[]

    TRAIN_abstract=[] #  abstract
    TEST_abstract=[]

    TRAIN_groundtruth=[] # 0 1 other
    TEST_groundtruth=[]

    TRAIN_journal=[] #    journal_if
    TEST_journal=[]

    TRAIN_author=[] #max_cited_by_count max_citation_count  max_document_count
    TEST_author=[]

    TRAIN_index=[] #i
    TEST_index=[]


    sum=0
    for iii in data_merge:
            sum+=1
            if sum<=1500:
                TRAIN_title.append([])
                TRAIN_abstract.append([])
                TRAIN_groundtruth.append([])
                TRAIN_journal.append([])
                TRAIN_author.append([])
                TRAIN_index.append([])


                for s in range(len(data_merge[iii])):
                    temp_index1 = None
                    temp_index2 = None
                    temp1=[]

                    temp = data_merge[iii][s]['title'].split()
                    temp1.append(temp)
                    if 'GENE' in temp:
                        temp_index1=find_all_index(temp,'GENE') #temp.index(i[0])
                    else:
                        temp_index1=-1
                    if 'TRAIT' in temp:
                        temp_index2 = find_all_index(temp, 'TRAIT')
                    else:
                        temp_index2=-1
                    temp1.append([0 for i in range(len(temp))])
                    if temp_index1!=-1:
                        for i_index in temp_index1:
                            temp1[1][i_index]=1
                    if temp_index2 != -1:
                        for i_index in temp_index2:
                            temp1[1][i_index] = 2
                    TRAIN_title[sum-1].append(temp1)


                    temp2=[]
                    temp = data_merge[iii][s]['abstract']
                    temp=temp.split()
                    temp2.append(temp)
                    if 'GENE' in temp:
                        temp_index1=find_all_index(temp,'GENE') #temp.index(i[0])
                    else:
                        temp_index1=-1
                    if 'TRAIT' in temp:
                        temp_index2 = find_all_index(temp, 'TRAIT')
                    else:
                        temp_index2=-1
                    temp2.append([0 for j in range(len(temp))])
                    if temp_index1!=-1:
                        for i_index in temp_index1:
                            # print temp_index1
                            # print "---"
                            temp2[1][i_index]=1
                    if temp_index2 != -1:
                        for i_index in temp_index2:
                            # print temp_index2
                            temp2[1][i_index] = 2
                    TRAIN_abstract[sum - 1].append(temp2)

                    temp = data_merge[iii][s]['journal_if']

                    TRAIN_journal[sum - 1].append(temp)
                    temp=[]
                    temp.append(data_merge[iii][s]['max_cited_by_count'] )
                    temp.append(data_merge[iii][s]['max_citation_count'])
                    temp.append(data_merge[iii][s]['max_document_count'])
                    TRAIN_author[sum - 1].append(temp)
                    TRAIN_index[sum-1].append(['GENE','TRAIT'])
                    try:
                        if data_merge[iii][s]['PMID'] in data_ground_truth[iii]:
                            TRAIN_groundtruth[sum - 1].append('1')
                        else:
                            TRAIN_groundtruth[sum - 1].append('0')
                    except:
                        TRAIN_groundtruth[sum - 1].append('0')

            else:
                TEST_title.append([])
                TEST_abstract.append([])
                TEST_groundtruth.append([])
                TEST_journal.append([])
                TEST_author.append([])
                TEST_index.append([])

                for s in range(len(data_merge[iii])):
                    # temp = data_merge[iii][s]['title'].split()
                    # TEST_title[sum - 701].append(temp)
                    # temp = data_merge[iii][s]['abstract'].split()
                    # TEST_abstract[sum - 701].append(temp)
                    temp_index1 = None
                    temp_index2 = None
                    temp1 = []

                    temp = data_merge[iii][s]['title'].split()
                    temp1.append(temp)
                    if 'GENE' in temp:
                        temp_index1 = find_all_index(temp, 'GENE')  # temp.index(i[0])
                    else:
                        temp_index1 = -1
                    if 'TRAIT' in temp:
                        temp_index2 = find_all_index(temp, 'TRAIT')
                    else:
                        temp_index2 = -1
                    temp1.append([0 for i in range(len(temp))])
                    if temp_index1 != -1:
                        for i_index in temp_index1:
                            temp1[1][i_index] = 1
                    if temp_index2 != -1:
                        for i_index in temp_index2:
                            temp1[1][i_index] = 2
                    TEST_title[sum - 1501].append(temp1)

                    temp2 = []
                    temp = data_merge[iii][s]['abstract']
                    temp = temp.split()
                    temp2.append(temp)
                    if 'GENE' in temp:
                        temp_index1 = find_all_index(temp, 'GENE')  # temp.index(i[0])
                    else:
                        temp_index1 = -1
                    if 'TRAIT' in temp:
                        temp_index2 = find_all_index(temp, 'TRAIT')
                    else:
                        temp_index2 = -1
                    temp2.append([0 for j in range(len(temp))])
                    if temp_index1 != -1:
                        for i_index in temp_index1:
                            # print temp_index1
                            # print "~~"
                            temp2[1][i_index] = 1
                    if temp_index2 != -1:
                        for i_index in temp_index2:
                            # print temp_index2
                            temp2[1][i_index] = 2
                    TEST_abstract[sum - 1501].append(temp2)

                    temp = data_merge[iii][s]['journal_if']
                    TEST_journal[sum - 1501].append(temp)
                    temp = []
                    temp.append(data_merge[iii][s]['max_cited_by_count'])
                    temp.append(data_merge[iii][s]['max_citation_count'])
                    temp.append(data_merge[iii][s]['max_document_count'])
                    TEST_author[sum - 1501].append(temp)
                    TEST_index[sum - 1501].append(['GENE','TRAIT'])
                    try:
                        if data_merge[iii][s]['PMID'] in data_ground_truth[iii]:
                            TEST_groundtruth[sum - 1501].append('1')
                        else:
                            TEST_groundtruth[sum - 1501].append('0')
                    except:
                        TEST_groundtruth[sum - 1501].append('0')
    print sum
    # print len(TEST_title[0])
    # print type(TEST_title[2])
    # print TEST_title[2]
    # print TEST_abstract[2]
    # print TEST_journal[2]
    # # print TEST_groundtruth
    # print TEST_author[2]
    #
    # print len(TRAIN_title[0])
    # print TRAIN_title[0][0]
    # print TRAIN_abstract[0][0]
    # print TRAIN_journal[0]
    # print TRAIN_groundtruth[0][0]
    # print TRAIN_author[0]
    # print TRAIN_index[0]
    # print TRAIN_author
    # #
    # print
    # print
    #
    # print len(TRAIN_title[2])
    # print TRAIN_title[2]
    # print TRAIN_abstract[2]
    # print TRAIN_journal[2]
    # print TRAIN_groundtruth
    # print TRAIN_author[2]
    #
    # print len(TEST_title)
    # print len(TRAIN_title)


    #vector=return_model(TRAIN_abstract,5)

    return TRAIN_abstract,TRAIN_author,TRAIN_groundtruth,TRAIN_journal,TRAIN_title,TRAIN_index, TEST_abstract,TEST_author,TEST_groundtruth,TEST_journal,TEST_title,TEST_index

if __name__ == '__main__':
    FOR_data()