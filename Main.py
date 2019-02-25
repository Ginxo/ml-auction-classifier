from MainHelper import MainHelper
from ml_algorithms.KMeansAlgorithm import KMeansAlgorithm
from ml_algorithms.KNearestAlgorithm import KNearestAlgorithm
from ml_algorithms.NaiveBayesAlgorithm import NaiveBayesAlgorithm

# Get tech and non-tech articles
tech_articles = MainHelper.get_tech_articles()
non_tech_articles = MainHelper.get_non_tech_articles()


#KMean
KMeansAlgorithm.run('https://www.cnet.com/news/galaxy-s10-plus-ongoing-review-whats-good-bad-so-far-samsung/', tech_articles, non_tech_articles)

#KNearest
KNearestAlgorithm.run('https://www.cnet.com/news/galaxy-s10-plus-ongoing-review-whats-good-bad-so-far-samsung/', tech_articles, non_tech_articles)

#Naive Bayes
NaiveBayesAlgorithm.run('https://www.cnet.com/news/galaxy-s10-plus-ongoing-review-whats-good-bad-so-far-samsung/', tech_articles, non_tech_articles)
