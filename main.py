import sys
import os
import time
import pickle
from os import path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

base = 'data'
review_data = 'movie_reviews'

def prepareData(foldNumber):
    fold = 'data/fold' + str(foldNumber)
    foldTrainFile = path.join(fold, 'train_file')
    foldTestFile = path.join(fold, 'test_file')
    foldTrainLabel = path.join(fold, 'train_label')
    foldTestLabel = path.join(fold, 'test_label')

    if not os.path.isdir(fold):
        os.makedirs(fold)

    data_dir = review_data
    classes = ['pos', 'neg']
    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:
                content = f.read()
                if fname.startswith('cv' + str(foldNumber)):
                    test_data.append(content)
                    test_labels.append(curr_class)
                else:
                    train_data.append(content)
                    train_labels.append(curr_class)

    with open(foldTrainFile, 'wb') as f:
        pickle.dump(train_data, f)
    with open(foldTestFile, 'wb') as f:
        pickle.dump(test_data, f)
    with open(foldTrainLabel, 'wb') as f:
        pickle.dump(train_labels, f)
    with open(foldTestLabel, 'wb') as f:
        pickle.dump(test_labels, f)

def doEvaluate(foldNumber):
    fold = 'data/fold' + str(foldNumber)
    foldTrainFile = path.join(fold, 'train_file')
    foldTestFile = path.join(fold, 'test_file')
    foldTrainLabel = path.join(fold, 'train_label')
    foldTestLabel = path.join(fold, 'test_label')

    with open(foldTrainFile, 'rb') as f:
        train_data = pickle.load(f)
    with open(foldTestFile, 'rb') as f:
        test_data = pickle.load(f)
    with open(foldTrainLabel, 'rb') as f:
        train_labels = pickle.load(f)
    with open(foldTestLabel, 'rb') as f:
        test_labels = pickle.load(f)

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # Perform classification with SVM, kernel=rbf
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1-t0
    time_liblinear_predict = t2-t1

    # Print results in a nice table
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))

def main():
    oper = -1
    while int(oper) != 0:
        print('**************************************')
        print('Choose one of the following: ')
        print('1 - Prepare Data')
        print('2 - Evaluate Data')
        print('0 - Exit')
        print('**************************************')
        oper = int(input("Enter your options: "))

        if oper == 0:
            exit()
        elif oper == 1:
            prepareData(0)
        elif oper == 2:
            doEvaluate(0)

if __name__ == "__main__":
    main()
