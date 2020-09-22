from math import log

import numpy as np
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from HW2.tfidf_base import preprocess, make_corpus

morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")

k = 2.0
b = 0.75


def bm25(word, document, doc_id, avgdl, tf_matrix) -> float:
    ld = len(document.split())
    N = len(tf_matrix)
    n_qi = len(tf_matrix[word].to_numpy().nonzero()[0])
    idf = log((N - n_qi + 0.5)/(n_qi + 0.5))
    score = idf * ((tf_matrix.iloc[doc_id][word] * (k + 1)) / (tf_matrix.iloc[doc_id][word] + k * (1 - b + ((b * ld) / avgdl))))
    return score


def make_tf(corpus):
    count_vectorizer = CountVectorizer()
    tf = count_vectorizer.fit_transform(corpus)
    tf_matrix = pd.DataFrame(tf.A,
                             columns=count_vectorizer.get_feature_names())
    return tf_matrix, count_vectorizer


def make_bm_matrix(tf_matrix, corpus, avgdl):
    bm_matrix = pd.DataFrame(columns=tf_matrix.columns)
    for row in tf_matrix.iterrows():
        idx = row[0]
        bm_matrix.loc[idx] = [bm25(word, corpus[idx], idx, avgdl, tf_matrix)
                              for word in tf_matrix.columns.values]
    return bm_matrix


def get_similar(query, vectorizer, bm_matrix, corpus):
    query_vec = vectorizer.transform(
        [' '.join(preprocess(query))]).toarray()
    result = bm_matrix.to_numpy().dot(query_vec.transpose())
    questions = []
    for i in np.argsort(result, axis=0)[::-1].transpose()[0]:
        questions.append(corpus[i])
    return questions


def main():
    all_q, clean = make_corpus('answers_base.xlsx')
    N = len(clean)
    lens = 0
    for text in clean:
        lens += len(text)
    avgdl = lens / N
    tf_matrix, vec = make_tf(clean)
    bm_matrix = make_bm_matrix(tf_matrix, clean, avgdl)
    query = input("Введите запрос: ")
    res = get_similar(query, vec, bm_matrix, all_q)
    print(res[:10])


if __name__ == "__main__":
    main()
