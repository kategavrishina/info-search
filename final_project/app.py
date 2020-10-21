import pickle
import pandas as pd
from flask import Flask, request, render_template
from final_project.get_answer import get_similar, get_classic_answer, \
    get_exp_answer

# загружаем векторайзеры и матрицы
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pickle', 'rb'))
count_vectorizer = pickle.load(open('count_vectorizer.pickle', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pickle', 'rb'))
bm25_matrix = pickle.load(open('bm25_matrix.pickle', 'rb'))
classic_matrix = pickle.load(open('classic_matrix.pickle', 'rb'))
exp_matrix = pickle.load(open('exp_matrix.pickle', 'rb'))

# загружаем сами данные и ответы
data = pd.read_excel('data_base.xlsx')
answers = pd.read_excel('ans_base.xlsx')


def search(query, search_method):
    if search_method == 'TF-IDF':
        search_result = get_similar(query, tfidf_vectorizer, tfidf_matrix.A,
                                    data, answers)
    elif search_method == 'BM25':
        search_result = get_similar(query, count_vectorizer, bm25_matrix,
                                    data, answers)
    elif search_method == 'word2vec_mean_vec':
        search_result = get_classic_answer(query, classic_matrix, data,
                                           answers)
    elif search_method == 'word2vec_matrix':
        search_result = get_exp_answer(query, exp_matrix, data, answers)
    else:
        raise TypeError('unsupported search method')

    return search_result


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.args:
        query = request.args['query']
        method = request.args['method']
        ans = search(query, method)[:5]
        return render_template('index.html', answer=ans)
    return render_template('index.html', answer=[])


if __name__ == '__main__':
    app.run(debug=False)
