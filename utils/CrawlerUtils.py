from domain.WebInfoFactory import WebInfoFactory
from utils.FileUtils import FileUtils
from utils.UrlUtils import UrlUtils


class CrawlerUtils(object):
    _CRAWLED_PAGES_CSV_FILE = "crawled_pages.csv"

    @staticmethod
    def scrape_source(url, magic_frags=['2019']):
        print('Scraping {} {} ...'.format(url, magic_frags))

        url_bodies = FileUtils.read_dictionary(CrawlerUtils._CRAWLED_PAGES_CSV_FILE)
        new_url_bodies = {}
        for link in (link for link in WebInfoFactory.url_to_web_info(url).get_links(magic_frags) if
                     link not in url_bodies):
            web_info = WebInfoFactory.url_to_web_info(link)
            new_url_bodies[link] = web_info.get_words()
            url_bodies[link] = new_url_bodies[link]
            print('{} scraped'.format(link))

        FileUtils.save_dictionary(new_url_bodies, CrawlerUtils._CRAWLED_PAGES_CSV_FILE, 'a')

        result = {k: v for k, v in url_bodies.items() if
                  UrlUtils.is_valid_url(k, url) and UrlUtils.contains_magic_frags(k, magic_frags)}
        print('{} {} articles {}.'.format(url, magic_frags, len(result)))
        return result
