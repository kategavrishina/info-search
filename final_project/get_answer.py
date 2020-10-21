import numpy as np
from final_project.create_matrices import normalize_vec, preprocess, \
    get_vector_mean, create_doc_matrix


# нахождение ближайших документов по матрицам Tfidf и BM25
def get_similar(query, vectorizer, matrix, data, answers):
    query_vec = vectorizer.transform([' '.join(preprocess(query))])
    result = query_vec.A.dot(matrix.T)[0]
    ans = []
    for i in np.argsort(result)[::-1]:
        num = data.loc[i]['Номер связки']
        text = answers['Текст ответа'].loc[answers['Номер связки'] ==
                                           num].values[0]
        if (num, text) not in ans:
            ans.append((num, text))
    return ans


# нахождение ближайших документов классическим методом w2v
def get_classic_answer(query, matrix, data, answers):
    vec = get_vector_mean(preprocess(query))
    res = normalize_vec(vec).dot(matrix.T)
    ans = []
    for i in np.argsort(res)[::-1]:
        num = data.loc[i]['Номер связки']
        text = answers['Текст ответа'].loc[answers['Номер связки'] == num].values[0]
        if (num, text) not in ans:
            ans.append((num, text))
    return ans


# нахождение ближайших документов экспериметальным методом w2v
def get_exp_answer(query, exp_matrix, data, answers, reduce_func=np.max):
    sims = []
    query_mat = create_doc_matrix(' '.join(preprocess(query)))
    for doc in exp_matrix:
        sim = doc.dot(query_mat.T)
        sim = reduce_func(sim, axis=0)
        sims.append(sim.sum())
    ans = []
    for i in np.argsort(sims)[::-1]:
        num = data.loc[i]['Номер связки']
        text = answers['Текст ответа'].loc[answers['Номер связки'] == num].values[0]
        if (num, text) not in ans:
            ans.append((num, text))
    return ans
