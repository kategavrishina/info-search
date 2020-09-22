import numpy as np
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")


def preprocess(f):
    sym = "0123456789.,?!…:;()[]-_|/\"'«»*{}<>@#$%^&№"
    s = []
    words = f.strip().lower().split()
    for word in words:
        word = word.strip(sym)
        if word != '':
            word = morph.parse(word)[0].normal_form
            if word not in russian_stopwords:
                s.append(word)
    return s


def make_corpus(file):
    df = pd.read_excel(file)
    corpus = []
    for row in df['Текст вопросов'].tolist():
        corpus += row.split('\n')
    clean_corpus = [' '.join(preprocess(text)) for text in corpus]
    return corpus, clean_corpus


def make_tfidf(corpus):
    vectorizer = TfidfVectorizer(stop_words=russian_stopwords)
    X = vectorizer.fit_transform(corpus)
    tfidf = pd.DataFrame(X.A, columns=vectorizer.get_feature_names())
    return tfidf, vectorizer


def get_similar(query, vectorizer, tfidf_matrix, corpus):
    query_tfidf = vectorizer.transform([' '.join(preprocess(query))]).toarray()
    result = tfidf_matrix.to_numpy().dot(query_tfidf.transpose())
    questions = []
    for i in np.argsort(result, axis=0)[::-1].transpose()[0]:
        questions.append(corpus[i])
    return questions


def main():
    all_q, clean = make_corpus('answers_base.xlsx')
    matrix, vec = make_tfidf(clean)
    query = input("Введите запрос: ")
    res = get_similar(query, vec, matrix, all_q)
    print(res[:10])


if __name__ == "__main__":
    main()
