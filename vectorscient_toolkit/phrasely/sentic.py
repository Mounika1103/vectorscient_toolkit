"""
The module provides wrappers to access to SenticNet conceptual information
database stored locally in XML format.

The database can be accessed online via following API:
http://sentic.net/api/

Or using PyPI package (which performs API queries under the hood):
https://pypi.python.org/pypi/senticnet
"""

from collections import defaultdict
import logging
import pathlib
import pickle
import glob
import sys
import os

try:
    from bs4 import BeautifulSoup
    LOAD_ONLY = False

except ImportError:
    LOAD_ONLY = True

    class BeautifulSoup:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError()


XMLNS_ATTR = 'xmlns=""'
SENTIC_PREFIX = 'http://sentic.net/api'
W3_PREFIX = 'http://www.w3.org/2001/xmlschema#'


def prepare_sentic_net_content(sentic_path: str,
                               folder: str, max_entries: int=1000):
    """
    Converts SenticNet RDF XML file into set of XML parsing trees.

    Args:
        sentic_path (str): the path to SenticNet concepts set
        folder (str): the folder to store parsing trees objects
        max_entries (int): the maximal quantity of words per tree
    """
    if LOAD_ONLY:
        logging.info("Cannot parse sentic corpus without "
                     "BeautifulSoup package installed")
        return

    sentic_path = os.path.expandvars(os.path.expanduser(sentic_path))

    with open(sentic_path) as fp:
        entry_count = 0
        chunks, current_chunk = [], []

        for line in fp:

            key = line.lower()
            key = key.replace('rdf:', '').strip()
            key = key.replace(SENTIC_PREFIX, '')
            key = key.replace(XMLNS_ATTR, '')
            key = key.replace(W3_PREFIX, '')

            if key.startswith('<rdf:rdf') or key.startswith('</rdf'):
                continue

            if not key:
                continue

            if key.replace('<', '').startswith('/description'):
                entry_count += 1

            current_chunk.append(line)

            if entry_count == max_entries:
                chunks.append(current_chunk)
                current_chunk = []
                entry_count = 0

        if entry_count != max_entries and current_chunk:
            chunks.append(current_chunk)

    soups = [BeautifulSoup("\n".join(ch)) for ch in chunks]

    path = pathlib.Path(folder)

    if path.exists() and not path.is_dir():
        logging.info("'{}' should be a directory".format(folder))
        return

    if not path.exists():
        path.mkdir()

    for i, soup in enumerate(soups, 1):
        prefix = str(i).rjust(2, '0')
        soup_file_name = "{}_soup.pickle".format(prefix)
        file_path = path.joinpath(soup_file_name)
        with open(file_path.as_posix(), 'wb') as fp:
            pickle.dump(soup, fp)

    logging.info("Parsed XML markup trees saved into '{}' folder".format(folder))


def load_sentic_net_chunks(folder: str):
    """
    Loads dumped XML parsers into memory.
    """
    base = os.path.expandvars(os.path.expanduser(folder))
    lookup = defaultdict(list)

    for pickled_tree in glob.glob(os.path.join(base, "*.pickle")):
        with open(pickled_tree, 'rb') as fp:
            tree = pickle.load(fp)
        first_letters = {tag.text[0] for tag in tree('text')}
        for letter in first_letters:
            lookup[letter].append(tree)

    return lookup


class SenticNetConnector:
    """
    Base class for connection to SenticNet corpus.

    The SenticNet could be accessed in different ways including local file
    parsing as well as remote API calls. The base class allows to abstract
    details of internal implementation.
    """
    CONCEPT_PARAMETERS = (
        'text', 'pleasantness', 'attention',
        'sensitivity', 'aptitude', 'polarity'
    )

    def concept(self, *con) -> dict:
        """
        Returns dictionary with concept scores
        """
        query = " ".join(list(con)) if len(con) > 1 else str(con[0])
        return self._concept(query)

    def _concept(self, query: str) -> dict:
        """
        Concept querying. Should be overridden with actual implementation.
        """
        return None

    def pleasantness(self, *con) -> float:
        return self._get_parameter('pleasantness', *con)

    def attention(self, *con) -> float:
        return self._get_parameter('attention', *con)

    def sensitivity(self, *con) -> float:
        return self._get_parameter('sensitivity', *con)

    def polarity(self, *con) -> float:
        return self._get_parameter('polarity', *con)

    def _get_parameter(self, key, *con):
        concept = self.concept(*con)
        if concept is None:
            return None
        return concept[key]


class SenticNetLocalStorage(SenticNetConnector):
    """
    Wrapper over locally stored SenticNet corpus. The corpus is distributed
    in RDF XML format and should be parsed in advance.
    """

    def __init__(self, pickle_folder: str):
        """
        Parameters:
            pickle_folder (str): the folder with pickled XML parsing trees
        """
        self._pickle_folder = pickle_folder
        # TODO: throw exception on empty lookup
        self._lookup = load_sentic_net_chunks(pickle_folder)
        self._cache = {}

    def _concept(self, query: str):
        if query in self._cache:
            return self._cache[query]

        tag = self._look_for_entry(query)

        if not tag:
            return None

        text_key, *scores = self.CONCEPT_PARAMETERS
        result = {text_key: tag.find(text_key).text}
        for key in scores:
            result[key] = float(tag.find(key).text)
        self._cache[query] = result

        return result

    def _look_for_entry(self, query):
        first_letter = query[0]
        trees = self._lookup[first_letter]
        for tree in trees:
            result = tree.find('text', text=query)
            if result:
                return result.parent
        return None


def convert_xml_to_pickles():
    rec_limit = sys.getrecursionlimit()
    # hack to save heavy BS objects (though sometimes fails anyway)
    sys.setrecursionlimit(1000000)
    prepare_sentic_net_content(
        "/Users/ck/senticnet-3.0/senticnet3.rdf.xml", "/Users/ck/sentic_net")
    sys.setrecursionlimit(rec_limit)
