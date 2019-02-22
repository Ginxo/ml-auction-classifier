import itertools

from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize

from utils.UrlUtils import UrlUtils


class WebInfo:
    soup = None

    def __init__(self, url, html):
        self.url = url
        self.html = html

    def get_url(self):
        return self.url

    def get_body(self, token='p'):
        return ' '.join(map(lambda p: p.text, self._get_soup().find_all(token)))

    def get_title(self):
        return self._get_soup().title.text

    def _get_soup(self):
        if self.soup is None:
            self.soup = BeautifulSoup(self.html, 'html.parser')
        return self.soup

    def get_words(self):
        return ' '.join(list(itertools.chain.from_iterable([word_tokenize(sentence.lower()) for sentence in sent_tokenize('{} {}'.format(self.get_title(), self.get_body()))])))

    def get_links(self, magic_frags=[]):
        if len(magic_frags) > 0:
            return set([UrlUtils.join_relative_url_with_parent_url(link, self.url) for link in self.get_links() if UrlUtils.contains_magic_frags(link, magic_frags)])
        else:
            return set([UrlUtils.join_relative_url_with_parent_url(link["href"], self.url) for link in self._get_soup().findAll('a', href=True) if
                        UrlUtils.is_valid_url(link["href"])])
