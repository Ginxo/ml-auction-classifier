from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from math import log
from collections import defaultdict
from heapq import nlargest

# Now for the frequency summarizer class - which we have encountered
# before. To quickly jog our memories - given an (title,article-body) tuple
# the frequency summarizer has easy ways to find the most 'important'
# sentences, and the most important words. How is 'important' defined?
# Important = most frequent, excluding 'stopwords' which are generic
# words like 'the' etc which can be ignored
class FrequencySummarizer:
    def __init__(self, min_cut=0.1, max_cut=0.9):
        self._min_cut = min_cut
        self._max_cut = max_cut
        self._stopwords = set(stopwords.words('english') +
                              list(punctuation) +
                              [u"'s", '"'])
    def _compute_frequencies(self, word_sent, customStopWords=None):
        freq = defaultdict(int)
        if customStopWords is None:
            stopwords = set(self._stopwords)
        else:
            stopwords = set(customStopWords).union(self._stopwords)
        for sentence in word_sent:
            for word in sentence:
                if word not in stopwords:
                    freq[word] += 1
        m = float(max(freq.values()))
        for word in list(freq.keys()):
            freq[word] = freq[word] / m
            if freq[word] >= self._max_cut or freq[word] <= self._min_cut:
                del freq[word]
        return freq

    def extractFeatures(self, article, n, customStopWords=None):
        text = article[0]
        title = article[1]
        sentences = sent_tokenize(text)
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        self._freq = self._compute_frequencies(word_sent, customStopWords)
        if n < 0:
            return nlargest(len(self._freq_keys()), self._freq, key=self._freq.get)
        else:
            return nlargest(n, self._freq, key=self._freq.get)
        # let's summarize what we did here.

    def extractRawFrequencies(self, article):
        text = article[0]
        title = article[1]
        sentences = sent_tokenize(text)
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        freq = defaultdict(int)
        for s in word_sent:
            for word in s:
                if word not in self._stopwords:
                    freq[word] += 1
        return freq

    def summarize(self, article, n):
        text = article[0]
        title = article[1]
        sentences = sent_tokenize(text)
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        self._freq = self._compute_frequencies(word_sent)
        ranking = defaultdict(int)
        for i, sentence in enumerate(word_sent):
            for word in sentence:
                if word in self._freq:
                    ranking[i] += self._freq[word]
        sentences_index = nlargest(n, ranking, key=ranking.get)

        return [sentences[j] for j in sentences_index]