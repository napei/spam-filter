"""
Spam detection ML Algorithm
Adapted from : https://www.kaggle.com/veleon/spam-classification/execution

@author: Nathaniel Peiffer
"""


from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.linear_model import *
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import *
from sklearn.naive_bayes import *
from sklearn.pipeline import Pipeline
from sklearn import datasets, svm, metrics, tree
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from collections import Counter, OrderedDict

from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *

import pandas as pd
import numpy as np
import os
from email.parser import BytesParser
import email.policy
from bs4 import BeautifulSoup
import urlextract
from tqdm import tqdm
from string import punctuation, digits
from itertools import chain, islice
import re
import time
import pprint
import gc

import nltk
nltk.download('punkt')

ham_dir = "data/spamassassin/ham"
spam_dir = "data/spamassassin/spam"

# ham_filenames = list(sorted(os.listdir(ham_dir)))
# spam_filenames = list(sorted(os.listdir(spam_dir)))


def load_single_email(path):
    with open(path, "rb") as f:
        return BytesParser(policy=email.policy.default).parse(f)


def load_folder_of_emails(path):
    return [load_single_email(os.path.join(path, f)) for f in (tqdm(sorted(os.listdir(path)), desc="Loading emails: {}".format(path)))]


print("Loading emails")
ham_emails = load_folder_of_emails(ham_dir)
spam_emails = load_folder_of_emails(spam_dir)

print("Loaded {} ham emails and {} spam emails".format(
    len(ham_emails), len(spam_emails)))


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()


def html_to_plain(email):
    try:
        soup = BeautifulSoup(email.get_content(), 'html.parser')
        return soup.text.replace('\n\n', '')
    except:
        return ""


def get_email_subject(email):
    try:
        sub = email.get("Subject")
        return sub
    except:
        return ""


def email_to_plain(email):
    struct = get_email_structure(email)
    for part in email.walk():
        partContentType = part.get_content_type()
        if partContentType not in ['text/plain', 'text/html']:
            continue
        try:
            partContent = part.get_content()
        except:
            partContent = str(part.get_payload())
        if partContentType == 'text/plain':
            return partContent
        else:
            return html_to_plain(part)


# - Strip email headers
# - Convert to lowercase
# - Remove punctuation
# - Replace urls with "URL"
# - Replace numbers with "NUMBER"
# - Perform Stemming (trim word endings with library)
class EmailToWords(BaseEstimator, TransformerMixin):
    def __init__(self, includeSubject=True, stripNumbers=True):
        self.url_extractor = urlextract.URLExtract()
        self.stemmer = nltk.PorterStemmer()
        self.includeSubject = includeSubject
        self.stripNumbers = stripNumbers

    def fit(self, X, y=None):
        return self

    # Transforms raw email into parseable text
    def transform(self, X, y=None):
        c = []
        regex = re.compile(r"[0-9]+")
        for email in tqdm(X, desc='Transforming emails'):
            text = email_to_plain(email)
            subject = str(get_email_subject(email)).lower()
            if subject is None:
                subject = ""
            if text is None:
                text = ""
            text += " " + subject

            urls = self.url_extractor.find_urls(text)
            for url in urls:
                text = text.replace(url, ' URL ')
            text = re.sub(regex, ' NUMBER ', text)
            text = text.translate(str.maketrans('', '', punctuation)).lower()
            # c.append(text)
            words = word_tokenize(text)
            # c.update([self.stemmer.stem(word) for word in words])
            c.append(" ".join([self.stemmer.stem(word) for word in words]))
        return c


process_emails = Pipeline([
    ("email_stemming", EmailToWords()),
], verbose=True)

# Transform Ham
processed_ham = process_emails.fit_transform(ham_emails)
if len(ham_emails) != len(processed_ham):
    raise ValueError("Ham not ham")

# Transform Spam
processed_spam = process_emails.fit_transform(spam_emails)
if len(spam_emails) != len(processed_spam):
    raise ValueError("Spam not spam")

# Concatenate X and y and label
X = np.array(processed_ham + processed_spam, dtype=object)
y = np.array([0] * len(processed_ham) + [1] * len(processed_spam))

test_classifiers = [
    LogisticRegression(solver="liblinear"),
    BernoulliNB(),
    RandomForestClassifier(n_estimators=100, n_jobs=-1),
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    CalibratedClassifierCV(),
    DummyClassifier(),
    PassiveAggressiveClassifier(),
    RidgeClassifier(),
    RidgeClassifierCV(),
    SGDClassifier(),
    OneVsRestClassifier(SVC(kernel='linear')),
    OneVsRestClassifier(LogisticRegression()),
    KNeighborsClassifier()
]

test_vectorizers = [
    CountVectorizer(),
    TfidfVectorizer(),
    HashingVectorizer()
]


def benchmark(cs, vs, X, y):
    results = ["classifier,vectorizer,train_time,test_time,p_score,r_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    for c in tqdm(cs):
        for v in vs:
            # Train classifier
            X_train_vec = v.fit_transform(X_train)
            X_test_vec = v.transform(X_test)

            start_train_time = time.time()
            c.fit(X_train_vec, y_train)
            end_train_time = time.time()

            # Test classifier
            start_test_time = time.time()
            y_test_predicted = c.predict(X_test_vec)
            p_score = precision_score(y_test, y_test_predicted)
            r_score = recall_score(y_test, y_test_predicted)
            end_test_time = time.time()

            results.append(
                ",".join([c.__class__.__name__, v.__class__.__name__, "{}".format(end_train_time-start_train_time), "{}".format(end_test_time-start_test_time), "{:.5f}".format(p_score), "{:.5f}".format(r_score)]))
            gc.collect()
    return results

# Benchmark lots of classifiers
# print("Benchmarking")
# res = benchmark(test_classifiers, test_vectorizers, X, y)

# for r in res:
#     print(r)


# Benchmarking determined that PassiveAggressiveClassifier with TfidfVectorizer is the best
# Run test of unknown sample
classifier = PassiveAggressiveClassifier()
vectorizer = TfidfVectorizer()

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

X_train_vec = vectorizer.fit_transform(X_train)

classifier.fit(X_train_vec, y_train)

# Load custom data
custom_test_data = load_folder_of_emails("data/custom/spam")
x_cust_test = process_emails.transform(custom_test_data)
x_vect_cust_test = vectorizer.transform(x_cust_test)
print(classifier.predict(x_vect_cust_test))
