"""
ITS Assignment 2
Spam detection ML Algorithm

@author: Nathaniel Peiffer - 101 603 798
"""

import gc
import os
import re
import time
from email import message_from_binary_file
from email.message import Message
from string import punctuation

import nltk
import numpy as np
import urlextract
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer,
                                             TfidfVectorizer)
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, RidgeClassifier,
                                  RidgeClassifierCV, SGDClassifier)
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

# Needed for nltk stemming and stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Directory names for data
ham_dir = "data/spamassassin/ham"
spam_dir = "data/spamassassin/spam"
custom_ham_dir = "data/custom/ham"
custom_spam_dir = "data/custom/spam"


def load_single_email(path: str) -> Message:
    """Loads a single email from a filepath

    Args:
        path (str): Filepath to load

    Returns:
        Message: Message class representation of that email file
    """
    return message_from_binary_file(open(path, "rb"))


def load_folder_of_emails(path: str) -> "list[Message]":
    """Loads a folder of emails given a filepath

    Returns:
        list[Message]: list of loaded messages
    """
    return [load_single_email(os.path.join(path, f))
            for f in (tqdm(sorted(os.listdir(path)), desc="Loading emails - {}".format(path)))]


def html_to_plain(e: Message) -> str:
    """Converts a message class body into a plaintext string

    Args:
        e (Message): Message to parse

    Returns:
        str: html string that was parsed
    """
    try:
        soup = BeautifulSoup(e.get_payload(), 'html.parser')
        return soup.text.replace('\n\n', '')
    except:
        return ""


def get_email_subject(e: Message) -> str:
    """Gets the subject line of an email message

    Args:
        e (Message): Message to parse

    Returns:
        str: Email subject
    """
    try:
        sub = e.get("Subject")
        return sub
    except:
        return ""


def email_to_plain(e: Message) -> str:
    """Converts an email message to plaintext, also
        parses html content and multipart content if present

    Args:
        e (Message): the message to parse

    Returns:
        str: plaintext representation of the message
    """
    content: "list[str]" = []
    for p in e.walk():
        current_text: str = ""
        if p.get_content_type() in ['text/plain', 'text/html']:
            try:
                current_text = p.get_payload()
            except:
                current_text = str(p.get_payload())

            if p.get_content_type() != 'text/plain':
                current_text = html_to_plain(p)
        content.append(current_text)
    return " ".join(content)


class EmailToWords(BaseEstimator, TransformerMixin):
    """Transformer which converts a list of email messages into
    a list of plaintext strings which are stemmed, and reduced to
    unique words only
    """

    def __init__(self, includeSubject=True, stripNumbers=True, stripStopWords=True):
        """Initialiser

        Args:
            includeSubject (bool, optional): Include the subject as well as the email body. Defaults to True.
            stripNumbers (bool, optional): Strip numbers from text, replacing them with "NUMBER". Defaults to True.
            stripStopWords (bool, optional): Strip stop words from the text. Defaults to True.
        """
        self.url_extractor = urlextract.URLExtract()
        self.stemmer = nltk.PorterStemmer()
        self.includeSubject = includeSubject
        self.stripNumbers = stripNumbers
        self.stripStopWords = stripStopWords

    def fit(self, X, y=None):
        return self

    def transform(self, X: "list[Message]", y=None) -> "list[str]":
        """Transforms a list of messages into a list of stemmed text

        Returns:
            list[str]: List of stemmed message strings
        """
        c: "list[str]" = []
        match_numbers = re.compile(r"[0-9]+")
        stops = set(stopwords.words("english"))
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
                text = text.replace(str(url), ' URL ')
            text = re.sub(match_numbers, ' NUMBER ', text)
            text = text.translate(str.maketrans('', '', punctuation)).lower()
            tokenized = word_tokenize(text, language="english")
            # Remove stopwords
            if self.stripStopWords:
                tokenized = [
                    token for token in tokenized if not token in stops]
            c.append(" ".join([self.stemmer.stem(word)
                               for word in tokenized]))
        return c


def benchmark(cs: list, vs: list, X, y, X_custom, y_custom) -> "list[str]":
    """Runs a benchmark of all provided classifiers and vectorisers

    Args:
        cs (list): List of classifiers to test
        vs (list): List of vectorisers that will each be tested per classifier
        X (Any): Input data for training, of which 80% will be used
        y (Any): Array of labels for the X
        X_custom (Any): Custom test data for measuring performance on unseen and unrelated data
        y_custom (Any): Array of labels for y_custom

    Returns:
        list[str]: Returns CSV data relating to the benchmark
    """
    results = [
        "classifier,vectorizer,train_time,test_time,p_score,r_score,f_score,score,custom_f_score"]

    # Split training dataset into training and testing
    # Random state keeps runs consistent, ish.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Pre-vecotrise data with selected vectorisers to avoid doing
    # this on every iteration of classifiers.
    local_vs: "list[tuple]" = []
    for vec in tqdm(vs, desc="vectorising to tuples"):
        tqdm.write("pre-vectorising with " + vec.__class__.__name__)
        # Vectorise data
        fitted = vec.fit(X)
        training_data = fitted.transform(X_train)
        testing_data = fitted.transform(X_test)
        custom_data = fitted.transform(X_custom)

        local_vs.append((training_data, testing_data,
                         custom_data, vec.__class__.__name__))

    print("Begin benchmark")
    for clf in tqdm(cs, desc="Classifiers"):
        for vec in tqdm(local_vs, desc="Vectorisers"):
            tqdm.write("doing "+clf.__class__.__name__ +
                       " - " + vec[3])
            try:
                # Get vectorised data
                training_data = vec[0]
                testing_data = vec[1]

                # Train classifier on dense data
                start_train_time = time.time()
                clf.fit(training_data, y_train)
                end_train_time = time.time()

                # Test with training data subset
                start_test_time = time.time()
                score = clf.score(testing_data, y_test)
                y_test_predicted = clf.predict(testing_data)
                p_score = precision_score(y_test, y_test_predicted)
                r_score = recall_score(y_test, y_test_predicted)
                f_score = f1_score(y_test, y_test_predicted)
                end_test_time = time.time()

                # Test with custom data
                X_custom_test = vec[2]
                custom_y_pred = clf.predict(X_custom_test)
                custom_f_score = f1_score(y_custom, custom_y_pred)

            except Exception as err:
                results.append(" ".join(
                    [clf.__class__.__name__, vec.__class__.__name__, "##ERROR## {}"]).format(err))
                print(err)
                continue
            results.append(
                ",".join([clf.__class__.__name__, vec[3],
                          "{:.5f}".format(end_train_time-start_train_time),
                          "{:.5f}".format(end_test_time-start_test_time),
                          "{:.5f}".format(p_score), "{:.5f}".format(r_score),
                          "{:.5f}".format(f_score), "{:.5f}".format(score),
                          "{:.5f}".format(custom_f_score)]))
            gc.collect()
    return results


print("Loading emails")
ham_emails = load_folder_of_emails(ham_dir)
spam_emails = load_folder_of_emails(spam_dir)
print("Loaded {} ham emails and {} spam emails. Total {}".format(
    len(ham_emails), len(spam_emails), len(ham_emails) + len(spam_emails)))
# Load custom test data
custom_ham_emails = load_folder_of_emails(custom_ham_dir)
custom_spam_emails = load_folder_of_emails(custom_spam_dir)
print("Loaded {} custom ham emails and {} custom spam emails. Total {}".format(
    len(custom_ham_emails), len(custom_spam_emails),
    len(custom_ham_emails) + len(custom_spam_emails)))

# Define email transform pipeline to save redeclaration
email_pipeline = Pipeline([('email_transform', EmailToWords())])

# Load and transform training dataset
processed_ham = email_pipeline.transform(ham_emails)
processed_spam = email_pipeline.transform(spam_emails)
# Load and transform custom test dataset
processed_custom_ham = email_pipeline.transform(custom_ham_emails)
processed_custom_spam = email_pipeline.transform(custom_spam_emails)

# Create training dataset
X = np.array(processed_ham + processed_spam, dtype=object)
y = np.array([0] * len(processed_ham) + [1] * len(processed_spam))

# Create custom dataset
X_custom = np.array(processed_custom_ham +
                    processed_custom_spam, dtype=object)
y_custom = np.array([0]*len(processed_custom_ham) +
                    [1]*len(processed_custom_spam))

test_classifiers = [
    SVC(kernel="linear", C=0.025, max_iter=1000),
    LinearSVC(),
    MultinomialNB(),
    BernoulliNB(),
    LogisticRegression(solver="liblinear", max_iter=1000),
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
    OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    KNeighborsClassifier()
]

max_vocab_size = 1000
# No max vocab size
test_vectorizers = [
    TfidfVectorizer(max_features=max_vocab_size),
    HashingVectorizer(n_features=max_vocab_size),
    CountVectorizer(max_features=max_vocab_size),
]

# Specify max vocab size
# test_vectorizers = [
#     TfidfVectorizer(max_features=1000),
#     HashingVectorizer(n_features=1000),
#     CountVectorizer(max_features=1000),
# ]

# Benchmark lots of classifiers
print("Benchmarking")
bench_start = time.time()
res = benchmark(test_classifiers, test_vectorizers, X, y, X_custom, y_custom)
res.append("Runtime: " + str(time.time() - bench_start))
print(res[-1])

for r in res:
    print(r)

f = open("data.csv", "w")
f.writelines(s + '\n' for s in res)
f.close()
