import logging
import urllib
from collections import defaultdict
from heapq import nlargest

import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from techclassifier.service.FrequencySummarizer import FrequencySummarizer
from techclassifier.utils.CrawlerUtils import CrawlerUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

urlWashingtonPostNonTech = "https://www.washingtonpost.com/sports"
urlNewYorkTimesNonTech = "https://www.nytimes.com/section/sports"
urlWashingtonPostTech = "https://www.washingtonpost.com/business/technology"
urlNewYorkTimesTech = "https://www.nytimes.com/section/technology"
washingtonPostTechArticles={}
washingtonPostNonTechArticles={}
newYorkTimesTechArticles={}
newYorkTimesNonTechArticles={}

logger.info('Getting Washingtonpost tech articles')
washingtonPostTechArticles = CrawlerUtils.scrape_source(urlWashingtonPostTech,
                                          ['2019', 'technology'])
logger.info('Washingtonpost tech articles {}'.format(len(washingtonPostTechArticles)))

logger.info('Getting Washingtonpost nont-tech articles')
washingtonPostNonTechArticles = CrawlerUtils.scrape_source(urlWashingtonPostNonTech,
                                             ['2019', 'sport'])
logger.info('Washingtonpost non-tech articles {}'.format(len(washingtonPostNonTechArticles)))



logger.info('Getting NYTimes tech articles')
newYorkTimesTechArticles = CrawlerUtils.scrape_source(urlNewYorkTimesTech,
                                        ['2019', 'technology'])
logger.info('NYTimes tech articles {}'.format(len(newYorkTimesTechArticles)))

logger.info('Getting NYTimes non-tech articles')
newYorkTimesNonTechArticles = CrawlerUtils.scrape_source(urlNewYorkTimesNonTech,
                                           ['2019', 'sport'],)
logger.info('NYTimes non-tech articles {}'.format(len(newYorkTimesNonTechArticles)))



# In[ ]:

# Now let's collect these article summaries in an easy to classify form
articleSummaries = {}
for techUrlDictionary in [newYorkTimesTechArticles, washingtonPostTechArticles]:
    for articleUrl in techUrlDictionary:
        if techUrlDictionary[articleUrl][0] is not None:
            if len(techUrlDictionary[articleUrl][0]) > 0:
                fs = FrequencySummarizer()
                summary = fs.extractFeatures(techUrlDictionary[articleUrl], 25)
                articleSummaries[articleUrl] = {'feature-vector': summary,
                                                'label': 'Tech'}
for nontechUrlDictionary in [newYorkTimesNonTechArticles, washingtonPostNonTechArticles]:
    for articleUrl in nontechUrlDictionary:
        if nontechUrlDictionary[articleUrl][0] is not None:
            if len(nontechUrlDictionary[articleUrl][0]) > 0:
                fs = FrequencySummarizer()
                summary = fs.extractFeatures(nontechUrlDictionary[articleUrl], 25)
                articleSummaries[articleUrl] = {'feature-vector': summary,
                                                'label': 'Non-Tech'}


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


testUrl = "http://doxydonkey.blogspot.in"
testArticle = getDoxyDonkeyText(testUrl, "post-body")

fs = FrequencySummarizer()
testArticleSummary = fs.extractFeatures(testArticle, 25)

# In[ ]:

similarities = {}
for articleUrl in articleSummaries:
    oneArticleSummary = articleSummaries[articleUrl]['feature-vector']
    similarities[articleUrl] = len(set(testArticleSummary).intersection(set(oneArticleSummary)))

labels = defaultdict(int)
knn = nlargest(5, similarities, key=similarities.get)
for oneNeighbor in knn:
    labels[articleSummaries[oneNeighbor]['label']] += 1

nlargest(1, labels, key=labels.get)

# In[ ]:

cumulativeRawFrequencies = {'Tech': defaultdict(int), 'Non-Tech': defaultdict(int)}
trainingData = {'Tech': newYorkTimesTechArticles, 'Non-Tech': newYorkTimesNonTechArticles}
for label in trainingData:
    for articleUrl in trainingData[label]:
        if len(trainingData[label][articleUrl][0]) > 0:
            fs = FrequencySummarizer()
            rawFrequencies = fs.extractRawFrequencies(trainingData[label][articleUrl])
            for word in rawFrequencies:
                cumulativeRawFrequencies[label][word] += rawFrequencies[word]

# In[ ]:

techiness = 1.0
nontechiness = 1.0
for word in testArticleSummary:
    # for each 'feature' of the test instance -
    if word in cumulativeRawFrequencies['Tech']:
        techiness *= 1e3 * cumulativeRawFrequencies['Tech'][word] / float(
            sum(cumulativeRawFrequencies['Tech'].values()))
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
    if word in cumulativeRawFrequencies['Non-Tech']:
        nontechiness *= 1e3 * cumulativeRawFrequencies['Non-Tech'][word] / float(
            sum(cumulativeRawFrequencies['Non-Tech'].values()))
        # we multiply the techiness by the probability of this word
        # appearing in a tech article (based on the training data)
    else:
        nontechiness /= 1e3

# we are almost done! Now we simply need to scale the techiness
# and non-techiness by the probabilities of overall techiness and
# non-techiness. THis is simply the number of words in the tech and
# non-tech articles respectively, as a proportion of the total number
# of words
techiness *= float(sum(cumulativeRawFrequencies['Tech'].values())) / (
            float(sum(cumulativeRawFrequencies['Tech'].values())) + float(
        sum(cumulativeRawFrequencies['Non-Tech'].values())))
nontechiness *= float(sum(cumulativeRawFrequencies['Non-Tech'].values())) / (
            float(sum(cumulativeRawFrequencies['Tech'].values())) + float(
        sum(cumulativeRawFrequencies['Non-Tech'].values())))
if techiness > nontechiness:
    label = 'Tech'
else:
    label = 'Non-Tech'
print(label, techiness, nontechiness)


# In[ ]:

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
    fs = FrequencySummarizer()
    summary = fs.extractFeatures((oneDocument, ""),
                                 100,
                                 [u"according", u"also", u"billion", u"like", u"new", u"one", u"year", u"first",
                                  u"last"])
    if cluster not in keywords:
        keywords[cluster] = set(summary)
    else:
        keywords[cluster] = keywords[cluster].intersection(set(summary))

# In[ ]: