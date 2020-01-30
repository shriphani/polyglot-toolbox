import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib
import numpy as np

import sys
import fasttext

np.random.seed(1991)

def cluster_posts(sents_f, model_f, prefix, K):

    model = fasttext.load_model(model_f)

    embeddings = []
    sentences = []
    with open(sents_f) as handle:
        for new_line in handle:
            if len(new_line.split()) < 5:
                continue
            sentences.append(new_line.strip())

    sentences = np.random.choice(sentences, 20000)

    for sentence in sentences:
        embeddings.append(model.get_sentence_vector(sentence))

    embeddings = np.array(embeddings)

    kmeans = KMeans(n_clusters=K, random_state=0).fit(embeddings)

    preds = kmeans.predict(embeddings)

    for i in range(K):
        dest = prefix + str(i) + '.txt'

        with open(dest, 'w') as handle:
            where_arrs = np.where(preds==i)[0]
            for pos in where_arrs:
                handle.write(sentences[pos])
                handle.write('\n')

    joblib.dump(kmeans, prefix + '_langid.joblib')