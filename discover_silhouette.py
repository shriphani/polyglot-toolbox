import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import joblib
import numpy as np

import sys
import fasttext

np.random.seed(1991)

def discover_silhouette(sents_f, model_f, prefix='', max_k=10):
    ks = []
    metrics = []

    model = fasttext.load_model(model_f)

    embeddings = []
    sentences = []

    # load sentences
    with open(sents_f) as handle:
        for new_line in handle:
            sentences.append(new_line.strip())

    # sample
    sentences = np.random.choice(sentences, 20000)

    # get document embeddings
    for sentence in sentences:
        embeddings.append(model.get_sentence_vector(sentence))

    embeddings = np.array(embeddings)

    # TSNE
    dimred = TSNE(n_jobs=4).fit_transform(embeddings)

    # This bit of code from the sklearn example

    for K in range(2, max_k):

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(embeddings) + (K + 1) * 10])


        kmeans = KMeans(n_clusters=K, random_state=0).fit(dimred)

        preds = kmeans.fit_predict(dimred)

        metric = silhouette_score(dimred, preds)

        ks.append(K)
        metrics.append(metric)

        sample_silhouette_values = silhouette_samples(dimred, preds)
        y_lower = 10
        for i in range(K):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                                            sample_silhouette_values[preds == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.viridis(float(i) / K)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples


        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=metric, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(preds.astype(float) / K)
        ax2.scatter(dimred[:, 0], dimred[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = kmeans.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                    s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % K),
                     fontsize=14, fontweight='bold')


        plt.savefig(prefix + str(i) + 'stuff.png')