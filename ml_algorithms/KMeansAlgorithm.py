from sklearn.cluster import KMeans
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
        X = vectorizer.fit_transform(document_corpus)  # each X is the vector representation of each document

        # A K-means cluster is created. 15 clusters, by k-means++, 200 iterations,
        # https://scikit-learn.org/stable/modules/clustering.html#k-means
        km = KMeans(n_clusters=10, init='k-means++', max_iter=200, n_init=1, verbose=True)
        km.fit(X)  # assign a label or cluster to each vector

        keyword_clusters = {}
        for i, cluster in enumerate(km.labels_):
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
