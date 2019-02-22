from utils.CrawlerUtils import CrawlerUtils


class MainHelper(object):

    @staticmethod
    def get_tech_articles():
        nyt_tech_articles = CrawlerUtils.scrape_source('https://www.nytimes.com/section/technology',
                                                       ['2019', 'technology'])
        wp_tech_articles = CrawlerUtils.scrape_source('https://www.washingtonpost.com/business/technology',
                                                      ['2019', 'technology'])
        return {**nyt_tech_articles, **wp_tech_articles}

    @staticmethod
    def get_non_tech_articles():
        nyt_non_tech_articles = CrawlerUtils.scrape_source('https://www.nytimes.com/section/sports', ['2019', 'sport'])
        wp_non_tech_articles = CrawlerUtils.scrape_source('https://www.washingtonpost.com/sports', ['2019', 'sport'])
        return {**nyt_non_tech_articles, **wp_non_tech_articles}
