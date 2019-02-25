import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from domain.TechArticlesConstants import ENGLISH_STOP_WORDS
from domain.WebInfoFactory import WebInfoFactory
from service.FrequencySummarizer import FrequencySummarizer


class KMeansAlgorithm(object):

    @staticmethod
    def run(url, tech_articles, non_tech_articles):
        print('-----------------------------------------')
        print('---------- K-Means algorithm --------')

        document_corpus = []
        for article in {**tech_articles, **non_tech_articles}.values():
            document_corpus.append(article)

        # returns a set of vectors that are weighted by th-idf
        vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=ENGLISH_STOP_WORDS)
        vectorized = vectorizer.fit_transform(document_corpus)  # each X is the vector representation of each document

        # A K-means cluster is created. 15 clusters, by k-means++, 200 iterations,
        # https://scikit-learn.org/stable/modules/clustering.html#k-means
        kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=200, n_init=1, verbose=True)
        kmeans.fit(vectorized)  # assign a label or cluster to each vector

        keyword_clusters = {}
        for i, cluster in enumerate(kmeans.labels_):
            document = document_corpus[i]
            summary = FrequencySummarizer().extract_features(document, 100)
            if len(summary) == 100:
                if cluster not in keyword_clusters:
                    keyword_clusters[cluster] = set(summary)
                else:
                    keyword_clusters[cluster] = keyword_clusters[cluster].intersection(set(summary))

        print('Keyword Clusters {}'.format(keyword_clusters))
        print('Document {} has this similarities {}'.format(url,
                                                            KMeansAlgorithm._get_similarities(url, keyword_clusters)))
        KMeansAlgorithm._plot(kmeans, vectorized)

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
    def _plot(kmeans, vectorized):
        labels_color_map = {
            0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
            5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
        }
        print(kmeans.cluster_centers_)
        pca_num_components = 2
        reduced_data = PCA(n_components=pca_num_components).fit_transform(vectorized.todense())
        fig, ax = plt.subplots()
        for index, instance in enumerate(reduced_data):
            pca_comp_1, pca_comp_2 = reduced_data[index]
            color = labels_color_map[kmeans.labels_[index]]
            ax.scatter(pca_comp_1, pca_comp_2, c=color)
        plt.show()
