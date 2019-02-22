import logging
import urllib
from requests import HTTPError

from domain.WebInfo import WebInfo


class WebInfoFactory(object):

    @staticmethod
    def url_to_web_info(url) -> WebInfo:
        try:
            with urllib.request.urlopen(url) as response:
                return WebInfo(url, response.read())
        except HTTPError as e:
            if e.code == 404:
                logging.getLogger(__name__).logger.error('{} not found'.format(url))
            else:
                logging.getLogger(__name__).logger.error('{} ERROR CODE: {}'.format(url, e.code))
        except UnicodeEncodeError as e:
            logging.getLogger(__name__).logger.error('{} Encoded Error: {}'.format(url, e))
        return None
