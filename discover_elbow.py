import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import joblib
import numpy as np

import sys
import fasttext

np.random.seed(1991)

def discover_elbow(sents_f, model_f, prefix, max_k):
    ks = []
    metrics = []

    model = fasttext.load_model(model_f)

    embeddings = []
    sentences = []
    with open(sents_f) as handle:
        for new_line in handle:
            sentences.append(new_line.strip())

    sentences = np.random.choice(sentences, 20000)

    for sentence in sentences:
        embeddings.append(model.get_sentence_vector(sentence))

    embeddings = np.array(embeddings)

    for K in range(2, max_k):

        kmeans = KMeans(n_clusters=K, random_state=0).fit(embeddings)

        preds = kmeans.predict(embeddings)

        metric = sum(np.min(cdist(embeddings, kmeans.cluster_centers_, 'euclidean'), axis=1)) / embeddings.shape[0]

        ks.append(K)
        metrics.append(metric)

    from matplotlib.pyplot import figure, show

    ax = figure().gca()

    ax.plot(ks, metrics, 'bx-')

    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Number of Clusters')
    plt.ylabel('k-Means Cluster Objective')
    plt.savefig(prefix + '.png')