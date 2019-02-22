from collections import defaultdict
from heapq import nlargest

from domain.TechArticlesConstants import FEATURE_VECTOR, LABEL, TECH_LABEL, NON_TECH_LABEL
from domain.WebInfoFactory import WebInfoFactory
from service.FrequencySummarizer import FrequencySummarizer
from utils.SummarizeUtils import SummarizeUtils


class KNearestAlgorithm(object):

    @staticmethod
    def run(url, tech_articles, non_tech_articles):
        print('-----------------------------------------')
        print('------------ KNearest algorithm ---------')
        article_summaries = KNearestAlgorithm._get_summary(tech_articles, non_tech_articles)
        similarities = KNearestAlgorithm._get_similarities(url, article_summaries)

        labels = defaultdict(int)
        k_nearest_neighbours = nlargest(5, similarities, key=similarities.get)
        for one_neighbour in k_nearest_neighbours:
            # how many tech or non-tech articles it is similar
            labels[article_summaries[one_neighbour][LABEL]] += 1
        print(
            'The article {} has these k-nearest neighbours articles {}, which are {}, mainly {}'.format(
                url,
                k_nearest_neighbours,
                labels,
                nlargest(1, labels,
                         key=labels.get)))

    @staticmethod
    def _get_summary(tech_articles, non_tech_articles):
        # Summarize tech and non-tech articles
        tech_summaries = SummarizeUtils.articles_sumarizator(tech_articles, TECH_LABEL)
        non_tech_summaries = SummarizeUtils.articles_sumarizator(non_tech_articles, NON_TECH_LABEL)
        return {**tech_summaries, **non_tech_summaries}

    @staticmethod
    def _get_similarities(url, article_summaries):
        result = {}
        article_to_check = WebInfoFactory.url_to_web_info(url).get_words()
        article_summary_to_check = FrequencySummarizer().extract_features(article_to_check, 25)

        for article_url in article_summaries:
            summary = article_summaries[article_url][FEATURE_VECTOR]
            # how many similar words they have in common
            result[article_url] = len(set(article_summary_to_check).intersection(set(summary)))

        return result
