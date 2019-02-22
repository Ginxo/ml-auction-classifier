from collections import defaultdict
from heapq import nlargest

from domain.TechArticlesConstants import FEATURE_VECTOR, LABEL
from domain.WebInfoFactory import WebInfoFactory
from service.FrequencySummarizer import FrequencySummarizer


class KNearestAlgorithm(object):

    @staticmethod
    def run(url, article_summaries):
        article_to_check = WebInfoFactory.url_to_web_info(url).get_words()
        article_summary_to_check = FrequencySummarizer().extract_features(article_to_check, 25)

        similarities = {}
        for article_url in article_summaries:
            summary = article_summaries[article_url][FEATURE_VECTOR]
            # how many similar words they have in common
            similarities[article_url] = len(set(article_summary_to_check).intersection(set(summary)))

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
