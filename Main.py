import logging
import urllib
from collections import defaultdict
from heapq import nlargest

import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from domain.TechArticlesConstants import FEATURE_VECTOR, LABEL
from domain.WebInfoFactory import WebInfoFactory
from ml_algorithms.KNearestAlgorithm import KNearestAlgorithm
from service.FrequencySummarizer import FrequencySummarizer
from utils.CrawlerUtils import CrawlerUtils
from utils.SummarizeUtils import SummarizeUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TECH_LABEL = 'Tech'
_NON_TECH_LABEL = 'Non-Tech'

nyt_tech_articles = CrawlerUtils.scrape_source('https://www.nytimes.com/section/technology', ['2019', 'technology'])
nyt_non_tech_articles = CrawlerUtils.scrape_source('https://www.nytimes.com/section/sports', ['2019', 'sport'])
wp_tech_articles = CrawlerUtils.scrape_source('https://www.washingtonpost.com/business/technology',
                                              ['2019', 'technology'])
wp_non_tech_articles = CrawlerUtils.scrape_source('https://www.washingtonpost.com/sports', ['2019', 'sport'])
tech_articles = {**nyt_tech_articles, **wp_tech_articles}
non_tech_articles = {**nyt_non_tech_articles, **wp_non_tech_articles}

# Now let's collect these article summaries in an easy to classify form
tech_summaries = SummarizeUtils.articles_sumarizator(tech_articles, _TECH_LABEL)
non_tech_summaries = SummarizeUtils.articles_sumarizator(non_tech_articles, _NON_TECH_LABEL)
article_summaries = {**tech_summaries, **non_tech_summaries}

#KNearest
KNearestAlgorithm.run('https://www.cnet.com/news/galaxy-s10-plus-ongoing-review-whats-good-bad-so-far-samsung/',
                      article_summaries)

#Naive Bayes
cumulativeRawFrequencies = {_TECH_LABEL: defaultdict(int), _NON_TECH_LABEL: defaultdict(int)}
trainingData = {_TECH_LABEL: tech_articles, _NON_TECH_LABEL: non_tech_articles}
for label in trainingData:
    for article_url in trainingData[label]:
        if len(trainingData[label][article_url][0]) > 0:
            frequencySummarizer = FrequencySummarizer()
            rawFrequencies = frequencySummarizer.extract_raw_frequencies(trainingData[label][article_url])
            for word in rawFrequencies:
                cumulativeRawFrequencies[label][word] += rawFrequencies[word]

# In[ ]:

article = WebInfoFactory.url_to_web_info('https://www.cnet.com/news/galaxy-s10-plus-ongoing-review-whats-good-bad-so-far-samsung/').get_words()
test_article_summary = FrequencySummarizer().extract_features(article, 25)
techiness = 1.0
nontechiness = 1.0
for word in test_article_summary:
    # for each 'feature' of the test instance -
    if word in cumulativeRawFrequencies[_TECH_LABEL]:
        techiness *= 1e3 * cumulativeRawFrequencies[_TECH_LABEL][word] / float(
            sum(cumulativeRawFrequencies[_TECH_LABEL].values()))
        # we multiply the techiness by the probability of this word
        # appearing in a tech article (based on the training data)
    else:
        techiness /= 1e3
        # THis is worth paying attention to. If the word does not appear
        # in the tech articles of the training data at all,we could simply
        # set that probability to zero - in fact doing so is the 'correct'
        # way mathematically, because that way all of the probabilities would
        # sum to 1. But that would lead to 'snap' decisions since the techiness
        # would instantaneously become 0. To prevent this, we decide to take
        # the probability as some very small number (here 1 in 1000, which is
        # actually not all that low)
    # Now the exact same deal- but for the nontechiness. We are intentionally
    # copy-pasting code (not a great software development practice) in order
    # to make the logic very clear. Ideally, we would have created a function
    # and called it twice rather than copy-pasting this code. In any event..
    if word in cumulativeRawFrequencies[_NON_TECH_LABEL]:
        nontechiness *= 1e3 * cumulativeRawFrequencies[_NON_TECH_LABEL][word] / float(
            sum(cumulativeRawFrequencies[_NON_TECH_LABEL].values()))
        # we multiply the techiness by the probability of this word
        # appearing in a tech article (based on the training data)
    else:
        nontechiness /= 1e3

# we are almost done! Now we simply need to scale the techiness
# and non-techiness by the probabilities of overall techiness and
# non-techiness. THis is simply the number of words in the tech and
# non-tech articles respectively, as a proportion of the total number
# of words
techiness *= float(sum(cumulativeRawFrequencies[_TECH_LABEL].values())) / (
        float(sum(cumulativeRawFrequencies[_TECH_LABEL].values())) + float(
    sum(cumulativeRawFrequencies[_NON_TECH_LABEL].values())))
nontechiness *= float(sum(cumulativeRawFrequencies[_NON_TECH_LABEL].values())) / (
        float(sum(cumulativeRawFrequencies[_TECH_LABEL].values())) + float(
    sum(cumulativeRawFrequencies[_NON_TECH_LABEL].values())))
if techiness > nontechiness:
    label = _TECH_LABEL
else:
    label = _NON_TECH_LABEL
print(label, techiness, nontechiness)


# In[ ]:

def getDoxyDonkeyText(testUrl, token):
    response = requests.get(testUrl)
    soup = BeautifulSoup(response.content)
    page = str(soup)
    title = soup.find("title").text
    mydivs = soup.findAll("div", {"class": token})
    text = ''.join(map(lambda p: p.text, mydivs))
    return text, title
    # our test instance, just like our training data, is nicely
    # setup as a (title,text) tuple

def getAllDoxyDonkeyPosts(url, links):
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    soup = BeautifulSoup(response)
    for a in soup.findAll('a'):
        try:
            url = a['href']
            title = a['title']
            if title == "Older Posts":
                print(title, url)
                links.append(url)
                getAllDoxyDonkeyPosts(url, links)
        except:
            title = ""
    return


# In[ ]:


blogUrl = "http://doxydonkey.blogspot.in"
links = []
getAllDoxyDonkeyPosts(blogUrl, links)
doxyDonkeyPosts = {}
for link in links:
    doxyDonkeyPosts[link] = getDoxyDonkeyText(link, 'post-body')

documentCorpus = []
for onePost in doxyDonkeyPosts.values():
    documentCorpus.append(onePost[0])

# In[ ]:

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = vectorizer.fit_transform(documentCorpus)
km = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(X)

keywords = {}
for i, cluster in enumerate(km.labels_):
    oneDocument = documentCorpus[i]
    frequencySummarizer = FrequencySummarizer()
    summary = frequencySummarizer.extract_features((oneDocument, ""),
                                                   100,
                                                   [u"according", u"also", u"billion", u"like", u"new", u"one", u"year",
                                                    u"first",
                                                    u"last"])
    if cluster not in keywords:
        keywords[cluster] = set(summary)
    else:
        keywords[cluster] = keywords[cluster].intersection(set(summary))

# In[ ]:
