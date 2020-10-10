"""
@author: Nathaniel Peiffer
"""
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from collections import Counter
import pandas as pd
import numpy as np
import os
from email.parser import BytesParser
import email.policy
from bs4 import BeautifulSoup
import urlextract

ham_dir = "data/hamnspam/ham"
spam_dir = "data/hamnspam/spam"

ham_filenames = [name for name in sorted(os.listdir(ham_dir))]
spam_filenames = [name for name in sorted(os.listdir(spam_dir))]


def load_email(path):
    with open(path, "rb") as f:
        return BytesParser(policy=email.policy.default).parse(f)


ham_emails = [load_email(os.path.join(ham_dir, name))
              for name in ham_filenames]
spam_emails = [load_email(os.path.join(spam_dir, name))
               for name in spam_filenames]


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
        return "empty"


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
    def __init__(self, stripHeaders=True, lowercaseConversion=True, punctuationRemoval=True,
                 urlReplacement=True, numberReplacement=True, stemming=True, includeSubject=True):
        self.stripHeaders = stripHeaders
        self.lowercaseConversion = lowercaseConversion
        self.punctuationRemoval = punctuationRemoval
        self.urlReplacement = urlReplacement
        self.url_extractor = urlextract.URLExtract()
        self.numberReplacement = numberReplacement
        self.stemming = stemming
        self.stemmer = nltk.PorterStemmer()
        self.includeSubject = includeSubject

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_to_words = []
        for email in X:
            text = email_to_plain(email)
            if text is None:
                text = 'empty'
            if self.lowercaseConversion:
                text = text.lower()

            if self.urlReplacement:
                urls = self.url_extractor.find_urls(text)
                for url in urls:
                    text = text.replace(url, 'URL')

            if self.punctuationRemoval:
                text = text.replace('.', '')
                text = text.replace(',', '')
                text = text.replace('!', '')
                text = text.replace('?', '')

            if self.includeSubject:
                s = get_email_subject(email)
                text = str(s).lower() + text

            word_counts = Counter(text.split())
            if self.stemming:
                stemmed_word_count = Counter()
                for word, count in word_counts.items():
                    stemmed_word = self.stemmer.stem(word)
                    stemmed_word_count[stemmed_word] += count
                word_counts = stemmed_word_count
            X_to_words.append(word_counts)
        return np.array(X_to_words)


class WordCountToVector(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_word_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_word_count[word] += min(count, 10)
        self.most_common = total_word_count.most_common()[
            :self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index,
                            (word, count) in enumerate(self.most_common)}
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


email_pipeline = Pipeline([
    ("Email to Words", EmailToWords()),
    ("Wordcount to Vector", WordCountToVector()),
])


# Training Model

X = np.array(ham_emails + spam_emails)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=85)
X_augmented_train = email_pipeline.fit_transform(X_train)

log_clf = LogisticRegression(solver="liblinear", random_state=85)
score = cross_val_score(log_clf, X_augmented_train, y_train, cv=3)

X_augmented_test = email_pipeline.transform(X_test)

log_clf = LogisticRegression(solver="liblinear", random_state=85)
log_clf.fit(X_augmented_train, y_train)

y_pred = log_clf.predict(X_augmented_test)

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))

custom_test_email = [load_email("data/custom/ham/gmail.1")]
x_cust_test = email_pipeline.transform(custom_test_email)

print(log_clf.predict(x_cust_test))
