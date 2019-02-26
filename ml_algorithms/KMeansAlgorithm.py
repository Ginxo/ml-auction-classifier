import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from domain.TechArticlesConstants import ENGLISH_STOP_WORDS, LABELS_COLOR_MAP
from domain.WebInfoFactory import WebInfoFactory
from service.FrequencySummarizer import FrequencySummarizer
from mpl_toolkits import mplot3d  # can't be unimported, the 3d projection need it


class KMeansAlgorithm(object):

    @staticmethod
    def run(url, tech_articles, non_tech_articles):
        print('-----------------------------------------')
        print('---------- K-Means algorithm --------')
        cluster_number = 5

        document_corpus = []
        for article in {**tech_articles, **non_tech_articles}.values():
            document_corpus.append(article)

        # returns a set of vectors that are weighted by th-idf
        vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=ENGLISH_STOP_WORDS)
        vectorized = vectorizer.fit_transform(document_corpus)  # each X is the vector representation of each document

        # A K-means cluster is created. 15 clusters, by k-means++, 200 iterations,
        # https://scikit-learn.org/stable/modules/clustering.html#k-means
        kmeans = KMeans(n_clusters=cluster_number, init='k-means++', max_iter=100, n_init=1, verbose=True)
        kmeans.fit(vectorized)  # assign a label or cluster to each vector

        KMeansAlgorithm._print_clusters(kmeans, vectorizer, cluster_number)

        prediction = KMeansAlgorithm._prediction(url, kmeans, vectorizer)
        print('The url {} is in the cluster {}'.format(url, prediction))
        KMeansAlgorithm._plot(kmeans, vectorized, prediction)

    @staticmethod
    def _get_similarities(url, keyword_clusters):
        result = {}
        article_to_check = WebInfoFactory.url_to_web_info(url).get_words()
        article_summary_to_check = FrequencySummarizer().extract_features(article_to_check, 25)

        for keyword_cluster in keyword_clusters:
            result[keyword_cluster] = '{}%'.format(
                KMeansAlgorithm._calculate_percentage(article_summary_to_check, keyword_clusters[keyword_cluster]))
        return result

    @staticmethod
    def _calculate_percentage(article_summary_to_check, keyword_cluster):
        return len(set(article_summary_to_check).intersection(keyword_cluster)) * 100 / len(article_summary_to_check)

    @staticmethod
    def _plot(kmeans, vectorized, prediction):

        pca_num_components = 3
        reduced_data = PCA(n_components=pca_num_components).fit_transform(vectorized.todense())
        ax = plt.axes(projection='3d')

        clusters = {}
        cluster_names = {}
        for index, instance in enumerate(reduced_data):
            pca_comp_1, pca_comp_2, pca_comp_3 = reduced_data[index]
            color = LABELS_COLOR_MAP[kmeans.labels_[index]]
            cluster = ax.scatter(pca_comp_1, pca_comp_2, pca_comp_3, c=color)
            clusters[kmeans.labels_[index]] = cluster
            cluster_names[kmeans.labels_[index]] = 'Cluster {}'.format(kmeans.labels_[index])

        print(kmeans.cluster_centers_)
        reduced_data = PCA(n_components=pca_num_components).fit_transform(kmeans.cluster_centers_)
        for index, instance in enumerate(reduced_data):
            pca_comp_1, pca_comp_2, pca_comp_3 = reduced_data[index]
            # the colours don't match with the label ones
            ax.scatter(pca_comp_1, pca_comp_2, pca_comp_3, c='r', marker='v')

        plt.title('K-means flipante ')
        plt.legend(clusters.values(), cluster_names.values())
        plt.xticks(())
        plt.yticks(())
        plt.show()

    @staticmethod
    def _print_clusters(kmeans, vectorizer, cluster_number):
        print('Top terms per cluster:')
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(cluster_number):
            print('Cluster {}'.format(i)),
            for ind in order_centroids[i, :10]:
                print('Cluster {}'.format(terms[ind])),
            print('')

    @staticmethod
    def _prediction(url, kmeans: KMeans, vectorizer: TfidfVectorizer):
        article_to_check = WebInfoFactory.url_to_web_info(url).get_words()
        article_summary_to_check = FrequencySummarizer().extract_features(article_to_check, 25)
        print('Prediction for these words {}'.format(article_summary_to_check))
        Y = vectorizer.transform([" ".join(str(x) for x in article_summary_to_check)])
        k_means_predicted = kmeans.predict(Y)
        return k_means_predicted
