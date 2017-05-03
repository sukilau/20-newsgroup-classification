from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np


# NB model
def NBModel(categories):
    # load data 
    twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

    # pipeline (tokenizer => transformer => MultinomialNB classifier)
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

    # evaluate on test set
    twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
    predicted = text_clf.predict(twenty_test.data)
    print("*** Naive Bayes Model ***")
    print("Newsgroup Categories : ", categories )
    print("Accuracy : {}%".format(np.mean(predicted == twenty_test.target)*100))
    print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
    print("Confusion Matrix : \n", metrics.confusion_matrix(twenty_test.target, predicted))
    
    
# linear SVM
def SVM(categories):
    # load data 
    twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

    # pipeline (tokenizer => transformer => linear SVM classifier)
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
    _ = text_clf.fit(twenty_train.data, twenty_train.target)

    # evaluate on test set
    twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
    predicted = text_clf.predict(twenty_test.data)
    print("*** SVM Model ***")
    print("Newsgroup Categories : ", categories )
    print("Accuracy : {}%".format(np.mean(predicted == twenty_test.target)*100)) 
    print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
    print("Confusion Matrix : \n", metrics.confusion_matrix(twenty_test.target, predicted))

    
# selected categories
categories = ["alt.atheism","soc.religion.christian","comp.graphics","sci.med"]
NBModel(categories)
SVM(categories)


# all categories
categories = ["comp.graphics","comp.os.ms-windows.misc","comp.sys.ibm.pc.hardware","comp.sys.mac.hardware","comp.windows.x","rec.autos","rec.motorcycles","rec.sport.baseball","rec.sport.hockey","sci.crypt","sci.electronics","sci.med","sci.space","misc.forsale","talk.politics.misc","talk.politics.guns","talk.politics.mideast","talk.religion.misc","alt.atheism","soc.religion.christian"]
NBModel(categories)
SVM(categories)
