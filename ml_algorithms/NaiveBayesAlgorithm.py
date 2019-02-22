from domain.TechArticlesConstants import TECH_LABEL, NON_TECH_LABEL
from domain.WebInfoFactory import WebInfoFactory
from service.FrequencySummarizer import FrequencySummarizer


class NaiveBayesAlgorithm(object):

    @staticmethod
    def run(url, tech_articles, non_tech_articles):
        print('-----------------------------------------')
        print('---------- Naive Bayes algorithm --------')

        word_frequencies = FrequencySummarizer.get_word_frequencies(
            {TECH_LABEL: tech_articles, NON_TECH_LABEL: non_tech_articles})
        (techiness, non_techiness) = NaiveBayesAlgorithm._get_probabilities(url, word_frequencies)

        techiness *= float(sum(word_frequencies[TECH_LABEL].values())) / (
                float(sum(word_frequencies[TECH_LABEL].values())) + float(
            sum(word_frequencies[NON_TECH_LABEL].values())))
        non_techiness *= float(sum(word_frequencies[NON_TECH_LABEL].values())) / (
                float(sum(word_frequencies[TECH_LABEL].values())) + float(
            sum(word_frequencies[NON_TECH_LABEL].values())))

        print(
            'The article {} is {} with tech probability of {} and non-tech probability of {}'.format(
                url,
                TECH_LABEL if techiness > non_techiness else NON_TECH_LABEL,
                techiness,
                non_techiness))

    @staticmethod
    def _get_probabilities(url, word_frequencies):
        test_article = WebInfoFactory.url_to_web_info(url).get_words()
        test_article_summary = FrequencySummarizer().extract_features(test_article, 25)

        techiness = 1.0
        non_techiness = 1.0
        for word in test_article_summary:
            if word in word_frequencies[TECH_LABEL]:
                techiness *= 1e3 * word_frequencies[TECH_LABEL][word] / float(
                    sum(word_frequencies[TECH_LABEL].values()))
            else:
                techiness /= 1e3
            if word in word_frequencies[NON_TECH_LABEL]:
                non_techiness *= 1e3 * word_frequencies[NON_TECH_LABEL][word] / float(
                    sum(word_frequencies[NON_TECH_LABEL].values()))
            else:
                non_techiness /= 1e3
        return techiness, non_techiness
