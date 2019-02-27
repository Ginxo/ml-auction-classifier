from MainHelper import MainHelper
from ml_algorithms.KMeansAlgorithm import KMeansAlgorithm
from ml_algorithms.KNearestAlgorithm import KNearestAlgorithm
from ml_algorithms.NaiveBayesAlgorithm import NaiveBayesAlgorithm

# Get tech and non-tech articles
tech_articles = MainHelper.get_tech_articles()
non_tech_articles = MainHelper.get_non_tech_articles()

# URL_TO_CHECK = 'https://www.cnet.com/news/galaxy-s10-plus-ongoing-review-whats-good-bad-so-far-samsung/'
URL_TO_CHECK = 'https://www.skysports.com/football/news/11095/11648594/john-terry-exclusive-i-havent-missed-playing-and-the-pressures-around-it'

# KNearest
KNearestAlgorithm.run(URL_TO_CHECK, tech_articles, non_tech_articles, 5)

# Naive Bayes
NaiveBayesAlgorithm.run(URL_TO_CHECK, tech_articles, non_tech_articles)

# KMean
KMeansAlgorithm.run(URL_TO_CHECK, tech_articles, non_tech_articles)
