from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup


class UrlUtils(object):

    @staticmethod
    def is_valid_url(url, parent_url=None):
        if "#" not in url and "mailto" not in url:
            if parent_url:
                url_netloc = urlparse(url).netloc
                parent_url_netloc = urlparse(parent_url).netloc
                return url_netloc in parent_url_netloc or parent_url_netloc in url_netloc
            else:
                return True
        return False

    @staticmethod
    def is_same_base_url(url, base_url):
        return urlparse(url).netloc == urlparse(base_url).netloc

    @staticmethod
    def contains_magic_frags(link, magic_frags=[]):
        for magic_frag in magic_frags:
            if magic_frag not in link:
                return False
        return True

    @staticmethod
    def join_relative_url_with_parent_url(url, parent_url):
        return urljoin(UrlUtils.get_base_url(parent_url), url)

    @staticmethod
    def get_base_url(url):
        return urlparse(url).scheme + "://" + urlparse(url).netloc

    @staticmethod
    def get_links(url, magic_frags=[]):
        if len(magic_frags) > 0:
            return set([link for link in UrlUtils.get_links() if UrlUtils.contains_magic_frags(link, magic_frags)])
        else:
            return set([link["href"] for link in BeautifulSoup(self.html, 'html.parser').findAll('a', href=True) if UrlUtils.is_valid_url(link["href"]) and UrlUtils.is_same_base_url(link["href"], UrlUtils.get_base_url(self.url))])

