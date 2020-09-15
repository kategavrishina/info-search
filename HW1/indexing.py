import json
import os
import random
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pymorphy2
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

morph = pymorphy2.MorphAnalyzer()


def preprocess(file):
    sym = "0123456789.,?!…:;()[]-_|/\"'«»*{}<>@#$%^&№"
    s = []
    for line in file:
        words = line.strip().lower().split()
        for word in words:
            word = word.strip(sym)
            if word != '':
                word = morph.parse(word)[0].normal_form
                s.append(word)
    return s


def index_json(files):
    d_json = defaultdict(list)
    for filename in tqdm(files):
        name = filename.split(os.sep)[-1]
        with open(filename, 'r', encoding='utf-8') as f:
            text = preprocess(f)
            c = Counter(text)
            for w in c.keys():
                d_json[w].append((name, c[w]))
    return d_json


def index_matrix(files):
    corpus = []

    for filename in tqdm(files):
        with open(filename, 'r', encoding='utf-8') as f:
            text = preprocess(f)
            corpus.append(' '.join(text))

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    return pd.DataFrame(X.toarray(), index=files,
                        columns=vectorizer.get_feature_names())


def get_answers(matrix):
    sum_all = matrix.sum()
    s = sum_all.sort_values()
    print("Какое слово является самым частотным?")
    print(s.tail(1).index.values[0], s.tail(1)[0])

    print("\nИ какое самым редким?")
    all_rare = s.where(s == 1).dropna()
    print("Их всего", len(all_rare), ", но вот некоторые из них:",
          random.sample(list(all_rare.index), 20))

    print("\nКакой набор слов есть во всех документах коллекции?")
    words = list(matrix.columns[(np.count_nonzero(matrix, axis=0) ==
                                 165).nonzero()])
    print(', '.join(words))

    print("\nКто из главных героев статистически самый популярный?")
    characters = {}
    for name in ['рэйчел', 'моника', 'фиби', 'джоуя', 'чендлер', 'росс']:
        characters[name] = sum_all[name]
    sorted_c = sorted(characters, key=characters.get, reverse=True)
    print(sorted_c[0], characters[sorted_c[0]])


def main():
    curr_dir = os.getcwd()
    dir = os.path.join(curr_dir, 'friends-data')

    files = []
    for r, d, f in os.walk(dir):
        if len(f) > 0:
            for file in f:
                files.append(os.path.join(r, file))

    d = index_json(files)
    m = index_matrix(files)

    get_answers(m)

    m.to_pickle('index_matrix.pkl')

    with open('index_json.json', 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
