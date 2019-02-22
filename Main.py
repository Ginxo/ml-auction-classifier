import urllib

import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from MainHelper import MainHelper
from ml_algorithms.KNearestAlgorithm import KNearestAlgorithm
from ml_algorithms.NaiveBayesAlgorithm import NaiveBayesAlgorithm
from service.FrequencySummarizer import FrequencySummarizer

# Get tech and non-tech articles
tech_articles = MainHelper.get_tech_articles()
non_tech_articles = MainHelper.get_non_tech_articles()

#KNearest
KNearestAlgorithm.run('https://www.cnet.com/news/galaxy-s10-plus-ongoing-review-whats-good-bad-so-far-samsung/', tech_articles, non_tech_articles)

#Naive Bayes
NaiveBayesAlgorithm.run('https://www.cnet.com/news/galaxy-s10-plus-ongoing-review-whats-good-bad-so-far-samsung/', tech_articles, non_tech_articles)

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
