import sys
import os
import time
import pickle
from os import path
import os

base = 'data'
review_data = 'movie_reviews'

code = {} #this is where code numbers for features is stored
ngram_freq = 5

def findFeature(data):
    unigrams = {}
    bigrams = {}

    for file_train in data:
        if '.DS_Store' in file_train:
            print ' discarded%s '%file_train
            data.remove(file_train)
        else:
            review = open(file_train, 'r').read()
            #split into words
            unis = review.split()
            for i in range(len(unis)):
                uni = unis[i]

                #add to bigrams dictionary
                if i < len(unis) - 1:
                    bi1 = unis[i].rsplit('/', 1)[0]
                    bi2 = unis[i+1].rsplit('/', 1)[0]
                    bi = bi1 + ' ' + bi2
                    bigrams[bi] = bigrams.get(bi, 0) + 1

                #add to unigrams dictionary
                word = uni.split('/', 1)[0] #split uni into pos tag and words(s)
                unigrams[word] = unigrams.get(word, 0) +1 #put word part into unigrams
    print 'Pairing Down Features'

    #Remove all unigram with frequency less than ngram_freq
    freq_unis = []
    for uni in unigrams.keys():
        if unigrams[uni] >= ngram_freq:
            freq_unis.append(uni)

    freq_unigrams = set(freq_unis)
    N = len(freq_unigrams)

    #Get top freq of Unigram
    M = int(round(len(freq_unigrams)/2))

    #get N most frequent bigrams
    bis = [(v, k) for k, v in bigrams.items()]
    bis.sort() #less frequent items are first
    bis.reverse() #more frequent items are first
    bis2 = bis[:N] #get the N most frequent bigrams
    freq_bis = []
    for bi in bis2:
        freq_bis.append(bi[1])
    freq_bigrams = set(freq_bis)

    #get M most frequent unigrams
    uni_pairs = [(v, k) for k, v in unigrams.items()]
    uni_pairs.sort() #less frequent items are first
    uni_pairs.reverse() #more frequent items are first
    uni_pairs2 = uni_pairs[:M] #get the N most frequent bigrams

    freq_unis = []
    for uni in uni_pairs2:
        freq_unis.append(uni[1])
    top_unigrams = set(freq_unis)

    #Make a dictionary of items with a unique number for each
    all_items = list(freq_unigrams) + list(freq_bigrams) + list(top_unigrams)
    i = 1
    for item in all_items:
        code[item] = i
        i +=1
    return (freq_unigrams, freq_bigrams, top_unigrams)

def doEvaluate(foldNumber):

    print "\n**************************************"
    print "Run on Fold" + str(foldNumber)
    print "**************************************\n"

    all_train = []
    all_test = []

    data_dir = review_data
    classes = ['pos', 'neg']
    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)
        for fname in os.listdir(dirname):
            if fname.startswith('cv' + str(foldNumber)):
                all_test.append(os.path.join(dirname, fname))
            else:
                all_train.append(os.path.join(dirname, fname))

    print 'Start Find Features'
    features = findFeature(all_train)

    print 'Number of unigrams features:%d ' %len(features[0])
    print 'Number of bigram features: %d'%len(features[1])
    print 'Number of top unigram features: %d'%len(features[2])

    feature_unigram_train = 'data/fold' + str(foldNumber) + '/train/feature_unigram'
    feature_bigram_train = 'data/fold' + str(foldNumber) + '/train/feature_bigram'
    feature_pos_train = 'data/fold' + str(foldNumber) + '/train/feature_pos'
    feature_top_unigram_train = 'data/fold' + str(foldNumber) + '/train/feature_top_unigram'
    feature_pos_unigram_train = 'data/fold' + str(foldNumber) + '/train/feature_pos_unigram'

    feature_unigram_test = 'data/fold' + str(foldNumber) + '/test/feature_unigram'
    feature_bigram_test = 'data/fold' + str(foldNumber) + '/test/feature_bigram'
    feature_pos_test = 'data/fold' + str(foldNumber) + '/test/feature_pos'
    feature_top_unigram_test = 'data/fold' + str(foldNumber) + '/test/feature_top_unigram'
    feature_pos_unigram_test = 'data/fold' + str(foldNumber) + '/test/feature_pos_unigram'

    testDir = 'data/fold' + str(foldNumber) + '/test/'
    if not os.path.isdir(testDir):
        os.makedirs(testDir)

    trainDir = 'data/fold' + str(foldNumber) + '/train/'
    if not os.path.isdir(trainDir):
        os.makedirs(trainDir)

    output_files_train = [feature_unigram_train,
                            feature_bigram_train,
                            feature_top_unigram_train]

    output_files_test = [feature_unigram_test,
                            feature_bigram_test,
                            feature_top_unigram_test]

    print 'Extract Features for Train Set'
    feature_extract(all_train, output_files_train, features)

    print 'Extract Features for Test Set'
    feature_extract(all_test, output_files_test, features)

def feature_extract(data, output_files, features):
    freq_unigrams = features[0]
    freq_bigrams = features[1]
    top_unigrams = features[2]

    train_unigrams = output_files[0]
    train_bigrams = output_files[1]
    train_top_unigrams = output_files[2]

    uni_features = []
    bi_features = []
    top_unis_features = []

    for file_train in data:
        review = open(file_train, 'r').read()
        unis = review.split()

        uni_L = []
        bi_L = []
        top_L = []
        for i in range(len(unis)):
            uni = unis[i]

            #extract the bigrams
            if i < len(unis) - 1:
                bi1 = unis[i].rsplit('/', 1)[0] #remove tag
                bi2 = unis[i+1].rsplit('/', 1)[0] #remove tag
                bi = bi1 + ' ' + bi2
                if bi in freq_bigrams:
                   bi_L.append(code[bi])

            #extract unigrams
            word = uni.split('/', 1)[0]
            if word in freq_unigrams:
                uni_L.append(code[word])

            #extract the top unigrams
            if word in top_unigrams:
               top_L.append(code[word])

        uni_L = list(set(uni_L))
        uni_L.sort()
        uni_s = ''
        for each in uni_L:
           uni_s += ' %d:1'%each

        bi_L = list(set(bi_L))
        bi_L.sort()
        bi_s = ''
        for each in bi_L:
           bi_s += ' %d:1'%each

        top_L = list(set(top_L))
        top_L.sort()
        top_s = ''
        for each in top_L:
           top_s += ' %d:1'%each

        uni_s += '\n'
        bi_s += '\n'
        top_s += '\n'

        if 'pos' in file_train: #it's a positive review
            uni_s = '+1' + uni_s
            bi_s = '+1' + bi_s
            top_s = '+1' + top_s
        else:
            uni_s = '-1' + uni_s
            bi_s = '-1' + bi_s
            top_s = '-1' +top_s

        uni_features.append(uni_s)
        bi_features.append(bi_s)
        top_unis_features.append(top_s)

    print 'Writing to output files'

    tr_unis = open(train_unigrams, 'w')
    for instance in uni_features:
       tr_unis.write(instance)
    tr_unis.close()

    tr_bis = open(train_bigrams, 'w')
    for instance in bi_features:
       tr_bis.write(instance)
    tr_bis.close()

    tr_top_unis = open(train_top_unigrams, 'w')
    for instance in top_unis_features:
       tr_top_unis.write(instance)
    tr_top_unis.close()

def getFileListInDir(rootdir):
   filelist = []
   def aux(junk, dirpath, namelist):
       for name in namelist:
           file = os.path.join(dirpath, name)
           if os.path.isdir(file) == False:
               filelist.append(file)
   os.path.walk(rootdir, aux, None)
   filelist = [fi.replace('\\','/') for fi in filelist]
   return filelist

def replaceTwoOrMore(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

def main():
    oper = -1
    while int(oper) != 0:
        print('**************************************')
        print('Choose one of the following: ')
        print('1 - Extract Features')
        print('0 - Exit')
        print('**************************************')
        oper = int(input("Enter your options: "))

        if oper == 0:
            exit()
        elif oper == 1:
            for i in range(0,10):
                doEvaluate(i)

if __name__ == "__main__":
    main()
