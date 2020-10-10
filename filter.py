"""
@author: Nathaniel Peiffer
"""


import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


def make_Dictionary(train_dir):
    print("Loading data")
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in tqdm(emails):
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words

    d = Counter(all_words)

    for item in tqdm(d.copy()):
        if item.isalpha() == False:
            d.pop(item)
        elif len(item) == 1:
            d.pop(item)
    d = d.most_common(3000)
    return d


def extract_features(mail_dir):
    print("Extracting Features")

    files = [open(os.path.join(mail_dir, fi)).read()
             for fi in os.listdir(mail_dir)]
    v = CountVectorizer()
    v.fit(files)
    vector = v.transform(files)

    # features_matrix = np.zeros((len(files), 3000))
    # docID = 0
    # for f in tqdm(files):
    #     with open(f) as fi:
    #         for i, line in enumerate(fi):
    #             if i == 2:
    #                 words = line.split()
    #                 for word in words:
    #                     wordID = 0
    #                     for i, d in enumerate(dictionary):
    #                         if d[0] == word:
    #                             wordID = i
    #                             features_matrix[docID,
    #                                             wordID] = words.count(word)
    #         docID = docID + 1
    # return features_matrix
    return vector.toarray()

# Create a dictionary of words with its frequency


train_dir = 'data/lingspam_lemm_stop/training'
dictionary = make_Dictionary(train_dir)


# Prepare feature vectors per training mail and its labels
print("Training")
train_labels = np.zeros(2602)
train_labels[2169:2601] = 1
train_matrix = extract_features(train_dir)

# Training SVM and Naive bayes classifier and its variants

model1 = LinearSVC(dual=True, max_iter=5000)
# model2 = MultinomialNB()

model1.fit(train_matrix, train_labels)
os._exit(1)
# model2.fit(train_matrix, train_labels)

# Test the unseen mails for Spam
print("Testing")
test_dir = 'data/lingspam_lemm_stop/testing'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(291)
test_labels[241:290] = 1

result1 = model1.predict(test_matrix)
# result2 = model2.predict(test_matrix)

print(confusion_matrix(test_labels, result1))
# print(confusion_matrix(test_labels, result2))
