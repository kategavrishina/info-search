import pickle
import re

import numpy as np
import pandas as pd
import pymorphy2
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")

model_file = 'araneum_none_fasttextcbow_300_5_2018.model'
model = KeyedVectors.load(model_file)


# препроцессинг
def preprocess(text):
    reg = re.compile('[^а-яА-Я ]')
    text = reg.sub('', text.lower().strip())
    words = text.strip().split()
    s = []
    for word in words:
        if word != '':
            word = morph.parse(word)[0].normal_form
            if word not in russian_stopwords:
                s.append(word)
    return s


# обработка данных
def create_data():
    ans = pd.read_excel('answers_base.xlsx')
    answers = pd.DataFrame(ans[['Номер связки', 'Текст ответа']])
    ans['Вопросы'] = ans['Текст вопросов'].apply(lambda text: text.split('\n'))
    ans = ans.explode('Вопросы')
    ans.reset_index(drop=True, inplace=True)
    ans.drop(['Текст вопросов', 'Текст ответа', 'Тематика'], axis=1, inplace=True)

    q = pd.read_excel('queries_base.xlsx', usecols='A, B', names=['Вопросы', 'Номер связки'])
    q = q[['Номер связки', 'Вопросы']]
    data = pd.concat([ans, q], ignore_index=True)
    data['Preprocessed'] = data['Вопросы'].apply(lambda text: ' '.join(preprocess(str(text))))
    data.dropna(inplace=True)
    data.drop(data[data.Preprocessed == ''].index, inplace=True)
    data = data.sample(int(len(data)*0.7)).reset_index(drop=True)

    return data, answers


# матрица tfidf из корпуса вопросов
def make_tfidf(corpus):
    vectorizer = TfidfVectorizer(stop_words=russian_stopwords)
    X = vectorizer.fit_transform(corpus)
    pickle.dump(vectorizer, open('tfidf_vectorizer.pickle', "wb"))
    return X


# подсчет значения bm25
def bm25(item, avgdl, ld, k=2.0, b=0.75) -> float:
    score = (item * (k + 1)) / (item + k * (1 - b + ((b * ld) / avgdl)))
    return score


# получение матрицы поэлементно
def get_bm(N, tf, avgdl, ld):
    bm = np.zeros((N, tf.shape[1]))
    for i in range(N):
        bm[i] = [bm25(word[i], avgdl, ld[i]) for word in tf.T]
    return bm


# преобразование корпуса вопросов в матрицу bm25
def bm_func(corpus):
    count_vectorizer = CountVectorizer()
    tf = count_vectorizer.fit_transform(corpus).A
    pickle.dump(count_vectorizer, open('count_vectorizer.pickle', 'wb'))

    N = len(tf)
    lens = 0
    ld = []
    for text in corpus:
        l = len(text.split())
        lens += l
        ld.append(l)
    avgdl = lens / N
    ld = np.array(ld)

    bm = get_bm(N, tf, avgdl, ld)

    return bm


# нормализация вектора
def normalize_vec(v):
    return v / np.sqrt(np.sum(v ** 2))


# усреднение семантических векторов слов в документе
def get_vector_mean(lemmas):
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    vec = np.zeros((model.vector_size,))

    for idx, lemma in enumerate(lemmas):
        try:
            lemmas_vectors[idx] = model[lemma]
        except AttributeError:
            continue

    if lemmas_vectors.shape[0] is not 0:
        vec = np.mean(lemmas_vectors, axis=0)

    if np.sum(vec ** 2) != 0:
        vec = normalize_vec(vec)

    return vec


# получение матрицы усредненных векторов для всего корпуса вопросов
def get_classic_matrix(corpus):
    matrix = np.zeros((len(corpus), model.vector_size))
    for idx, text in enumerate(corpus):
        matrix[idx] = get_vector_mean(text.split())
    return matrix


# получение матрицы семантических векторов слов для документа
def create_doc_matrix(text):
    lemmas = text.split()
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))

    for idx, lemma in enumerate(lemmas):
        lemmas_vectors[idx] = normalize_vec(model[lemma])

    return lemmas_vectors


# объединение всех матриц документов всего корпуса
def get_exp_matrix(corpus):
    list_of_mtx = []
    for idx, text in enumerate(corpus):
        list_of_mtx.append(create_doc_matrix(text))
    return list_of_mtx


def main():
    data, answers = create_data()
    data.to_excel('data_base.xlsx')
    answers.to_excel('ans_base.xlsx')
    corpus = data.Preprocessed
    tfidf_matrix = make_tfidf(corpus)
    pickle.dump(tfidf_matrix, open('tfidf_matrix.pickle', 'wb'))
    bm25_matrix = bm_func(corpus)
    pickle.dump(bm25_matrix, open('bm25_matrix.pickle', 'wb'))
    classic_matrix = get_classic_matrix(corpus)
    pickle.dump(classic_matrix, open('classic_matrix.pickle', 'wb'))
    exp_matrix = get_exp_matrix(corpus)
    pickle.dump(exp_matrix, open('exp_matrix.pickle', 'wb'))


if __name__ == "__main__":
    main()
