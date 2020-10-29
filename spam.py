"""
Spam detection ML Algorithm
Adapted from : https://www.kaggle.com/veleon/spam-classification/execution

@author: Nathaniel Peiffer
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from email.message import EmailMessage, MIMEPart, Message
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

import numpy as np
import os
import email

from email.parser import BytesParser
from email import message_from_binary_file
import email.policy
from bs4 import BeautifulSoup
import urlextract
from tqdm import tqdm
from string import punctuation
import re
import time
import gc

import nltk
nltk.download('punkt')

ham_dir = "data/spamassassin/ham"
spam_dir = "data/spamassassin/spam"

# ham_filenames = list(sorted(os.listdir(ham_dir)))
# spam_filenames = list(sorted(os.listdir(spam_dir)))


def load_single_email(path: str) -> Message:
    return message_from_binary_file(open(path, "rb"))


def load_folder_of_emails(path: str) -> "list[Message]":
    return [load_single_email(os.path.join(path, f)) for f in (tqdm(sorted(os.listdir(path)[:100]), desc="Loading emails: {}".format(path)))]


print("Loading emails")
ham_emails = load_folder_of_emails(ham_dir)
spam_emails = load_folder_of_emails(spam_dir)

print("Loaded {} ham emails and {} spam emails".format(
    len(ham_emails), len(spam_emails)))


def get_email_structure(e: Message):
    if isinstance(e, str):
        return e
    payload = e.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return e.get_content_type()


def html_to_plain(e: Message):
    try:
        soup = BeautifulSoup(e.get_payload(), 'html.parser')
        return soup.text.replace('\n\n', '')
    except:
        return ""


def get_email_subject(e: Message):
    try:
        sub = e.get("Subject")
        return sub
    except:
        return ""


def email_to_plain(e: Message):
    struct = get_email_structure(e)
    for p in e.walk():
        if p.get_content_type() in ['text/plain', 'text/html']:
            try:
                content = p.get_payload()
            except:
                content = str(p.get_payload())

            if p.get_content_type() == 'text/plain':
                return content
            else:
                return html_to_plain(p)

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
    def transform(self, X: "list[Message]", y=None):
        c: "list[str]" = []
        regex = re.compile(r"[0-9]+")
        e: Message
        for e in tqdm(X, desc='Transforming emails'):
            text = email_to_plain(e)
            subject = str(get_email_subject(e)).lower()
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
            words = word_tokenize(text)
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
    SVC(kernel="linear", C=0.025),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    MLPClassifier(alpha=1, max_iter=1000),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),

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
    OneVsRestClassifier(LogisticRegression()),
    KNeighborsClassifier()
]

test_vectorizers = [
    TfidfVectorizer(),
    CountVectorizer(),
    HashingVectorizer()
]


def benchmark(cs, vs, X, y):
    results = [
        "classifier,vectorizer,train_time,test_time,p_score,r_score,f_score,score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for c in tqdm(cs):
        for v in vs:
            try:
                # Train classifier on dense data
                X_train = v.fit_transform(X_train).todense()
                X_test = v.transform(X_test).todense()

                start_train_time = time.time()
                c.fit(X_train, y_train)
                end_train_time = time.time()

                # Test classifier
                start_test_time = time.time()
                score = c.score(X_test, y_test)
                y_test_predicted = c.predict(X_test)
                p_score = precision_score(y_test, y_test_predicted)
                r_score = recall_score(y_test, y_test_predicted)
                f_score = f1_score(y_test, y_test_predicted)
                end_test_time = time.time()

            except:
                results.append("##ERROR##")
                continue
            results.append(
                ",".join([c.__class__.__name__, v.__class__.__name__,
                          "{}".format(end_train_time-start_train_time),
                          "{}".format(end_test_time-start_test_time),
                          "{:.5f}".format(p_score), "{:.5f}".format(r_score), "{:.5f}".format(f_score), "{:.5f}".format(score)]))
            gc.collect()
    return results


# Benchmark lots of classifiers
print("Benchmarking")
bench_start = time.time()
res = benchmark(test_classifiers, test_vectorizers, X, y)

for r in res:
    print(r)

f = open("data.csv", "w")
f.writelines(s + '\n' for s in res)
f.close()
print(time.time() - bench_start)

# # Benchmarking determined that PassiveAggressiveClassifier with TfidfVectorizer is the best
# # Run test of unknown sample
# classifier = PassiveAggressiveClassifier()
# vectorizer = TfidfVectorizer()

# # Train classifier
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2)

# X_train_vec = vectorizer.fit_transform(X_train)

# classifier.fit(X_train_vec, y_train)

# # Load custom data
# custom_test_data = load_folder_of_emails("data/custom/spam")
# x_cust_test = process_emails.transform(custom_test_data)
# x_vect_cust_test = vectorizer.transform(x_cust_test)

# print(classifier.predict(x_vect_cust_test))
