from domain.TechArticlesConstants import FEATURE_VECTOR, LABEL
from service.FrequencySummarizer import FrequencySummarizer


class SummarizeUtils(object):

    @staticmethod
    def articles_sumarizator(articles, label):
        result = {}
        for article_url in (article_url for article_url in articles if articles[article_url][0] is not None ):
            if articles[article_url][0] is not None:
                if len(articles[article_url][0]) > 0:
                    result[article_url] = SummarizeUtils.article_sumarizator(articles[article_url], label)
        return result

    @staticmethod
    def article_sumarizator(article, label):
        frequency_summarizer = FrequencySummarizer()
        summary = frequency_summarizer.extract_features(article, 25)
        return {FEATURE_VECTOR: summary, LABEL: label}
