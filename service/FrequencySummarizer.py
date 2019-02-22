from collections import defaultdict
from heapq import nlargest
from string import punctuation

from nltk.corpus import stopwords

class FrequencySummarizer:
    def __init__(self, max_cut=0.95, min_cut=0.1):
        self._max_cut = max_cut
        self._min_cut = min_cut
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ["'s", '"', "'", "", '—', '’', "”", "“", '``'])

    def extract_features(self, article, n, customStopWords=None):
        self._freq = self._compute_frequencies(article, customStopWords)
        if n < 0:
            return nlargest(len(self._freq_keys()), self._freq, key=self._freq.get)
        else:
            return nlargest(n, self._freq, key=self._freq.get)

    def extract_raw_frequencies(self, article):
        freq = defaultdict(int)
        for word in (word for word in article.split() if word not in self._stopwords):
            freq[word] += 1
        return freq

    def _compute_frequencies(self, article, custom_stop_words=None):
        freq = defaultdict(float)
        stop_words = (set(self._stopwords) if custom_stop_words is None else set(custom_stop_words).union(self._stopwords))
        for word in (word for word in article.split() if word not in stop_words):
            freq[word] += 1
        m = float(max(freq.values()))
        for word in list(freq.keys()):
            freq[word] = freq[word] / m
            if freq[word] >= self._max_cut or freq[word] <= self._min_cut:
                del freq[word]
        return freq
