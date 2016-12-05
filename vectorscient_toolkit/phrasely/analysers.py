from string import ascii_lowercase
from enum import IntEnum
import math
import csv
import re

from sklearn.preprocessing import normalize
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
from nltk.util import ngrams
import numpy as np
import nltk

from .preprocessors import LemmasNormalizer
from .sentic import SenticNetConnector
from ..config import NLTK_DATA_PATH


nltk.data.path.append(NLTK_DATA_PATH)


class SentimentWeights:
    """
    Sentiment weights mapping. Default implementation just delegates its work
    to underlying dictionary. But probably more advanced search needed if
    words list is quite long.
    """

    def __init__(self, mapping):
        self._mapping = mapping

    def get_weight(self, key):
        return self._mapping.get(key, 0)

    @property
    def keys(self):
        return self._mapping.keys()


class SentimentResult(IntEnum):
    """
    Enumeration of verbose results for sentiment analyser.
    """
    VeryNegative = 0
    Negative = 1
    SomewhatNegative = 2
    Neutral = 3
    SomewhatPositive = 4
    Positive = 5
    VeryPositive = 6


DEFAULT_SENTIMENT_LEVELS = {
    -5: SentimentResult.VeryNegative,
    -4: SentimentResult.VeryNegative,
    -3: SentimentResult.Negative,
    -2: SentimentResult.Negative,
    -1: SentimentResult.SomewhatNegative,
    +0: SentimentResult.Neutral,
    +1: SentimentResult.SomewhatPositive,
    +2: SentimentResult.SomewhatPositive,
    +3: SentimentResult.Positive,
    +4: SentimentResult.Positive,
    +5: SentimentResult.VeryPositive
}


class SentimentAnalyser:
    """
    Base class for sentiment analysing algorithms. Implements sentiment mapping
    from numeric score to verbose representation and non-ascii symbols removing
    functionality.
    """

    def __init__(self, **params):
        """
        Parameters:
            sentiment_table (dict): mapping from sentiment weight to
                human-readable sentiment description
        """
        self._sentiment_table = params.get(
            "sentiment_table", DEFAULT_SENTIMENT_LEVELS)

    @classmethod
    def name(cls) -> str:
        """
        Human-readable analyser name.
        """
        return "SentimentAnalyser"

    def sentiment(self, text: str) -> float:
        """
        Stub for sentiment score calculation.
        """
        return 0.0

    def verbose(self, score: float) -> str:
        """
        Converts sentiment score into more verbose representation (i.e.
        'Neutral', 'Positive', etc.).

        Args:
            score (float): sentiment score assigned to sentence
        """
        thresholds = list(self._sentiment_table.keys())
        lower, upper = min(thresholds), max(thresholds)
        normalized_score = round(min(upper, max(lower, score)), ndigits=0)
        emotion = self._sentiment_table[normalized_score]
        return emotion.name

    @staticmethod
    def _remove_non_ascii_symbols(word):
        symbols = [letter for letter in word if letter in ascii_lowercase]
        cleaned = "".join(symbols)
        return cleaned

    def _convert_score_to_range(self, score: float):
        score_values = list(self._sentiment_table.keys())
        lo, hi = min(score_values), max(score_values)
        normalized = (hi if score > 0 else lo) * abs(score)
        return normalized


class SimpleSentimentAnalyser(SentimentAnalyser, LemmasNormalizer):
    """
    Implements simple sentiment analysis by lookup files with predefined words
    and phrases that assign numeric score to them.
    """

    PATTERN_SPLIT = re.compile(r'\W+')

    def __init__(self, **params):
        """
        Parameters:
            words_path (str): path to a file with words sentiment scores

            sentences_path (str): path to a file with sentences sentiment scores

            calc_ngram_size (bool): if true, then maximum n-gram size should be
                defined from size of the longest phrase stored in sentences file

        Raises:
            ValueError: sentiment weights files are not provided
        """
        super(SimpleSentimentAnalyser, self).__init__(**params)

        self._words_path = params.get("words_path", "")
        self._sentences_path = params.get("sentences_path", "")
        self._calc_ngram_size = params.get("calc_ngram_size", True)

        if not self._sentences_path or not self._words_path:
            raise ValueError("cannot init analyser without sentiment weights")

        self._words = self._read_weights(self._words_path)
        self._sentences = self._read_weights(self._sentences_path)
        self._longest_entry = None

        if self._calc_ngram_size:
            key_lengths = [len(k.split()) for k in self._sentences.keys]
            self._longest_entry = max(key_lengths)

        self._sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self._lemmatizer = nltk.WordNetLemmatizer()

    @classmethod
    def name(cls) -> str:
        return "Simple"

    @staticmethod
    def _read_weights(file_name: str) -> SentimentWeights:
        with open(file_name, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            mapping = {}
            for row in reader:
                if not row or len(row) != 2:
                    continue
                item, weight = row
                mapping[item.lower()] = float(weight)
        weights = SentimentWeights(mapping)
        return weights

    @property
    def ngram_size(self):
        return self._longest_entry

    @ngram_size.setter
    def ngram_size(self, n):
        """
        Allows to manually set n-gram size.
        """
        self._longest_entry = n

    def sentiment(self, text: str) -> float:
        """
        Estimates sentiment score for provided text.

        Args:
            text (str): sentence to be analysed
        """
        # normalized = text.lower().strip()
        # sentences = self._sent_detector.tokenize(normalized)
        sentences = self.normalize(text)
        scores = [self.sentence_sentiment(s) for s in sentences]
        if not sentences:
            return 0.0
        total_text_score = sum(scores)/math.sqrt(len(sentences))
        return total_text_score

    def sentence_sentiment(self, sentence: list) -> float:
        n = self.ngram_size
        max_score = 0.0

        for score in self.ngram_sentiment(sentence, max_n=n):
            if score == 0.0:
                continue
            max_score = max(max_score, score)

        if max_score > 0:
            return max_score

        score = self.word_sentiment(sentence)
        return score

    def word_sentiment(self, words: list) -> float:
        """
        Calculates sentiment score analysing each word separately.

        Args:
            words (list): the sentence words to be analysed
        """
        ascii_only = [self._remove_non_ascii_symbols(w) for w in words]

        lm = self._lemmatizer
        lemmas = [lm.lemmatize(w) for w in ascii_only if w]
        score = self._get_sentiment_score(lemmas, self._words)

        return score

    def ngram_sentiment(self, words: list, max_n: int=8):
        """
        Generates list of sentiment scores for different sizes of n-grams.

        Args:
            words (str): the list of words in sentence
            max_n (int): maximal size of n-gram to be tried
        """
        for n in range(1, max_n + 1):
            terms = [" ".join(ngram) for ngram in ngrams(words, n)]
            if not terms:
                yield 0.0
            else:
                sentiments = [self._sentences.get_weight(t) for t in terms]
                sentiment_score = max(sentiments)
                yield sentiment_score

    @staticmethod
    def _get_sentiment_score(terms, weights):
        sentiments = [weights.get_weight(w) for w in terms]
        if not sentiments:
            return 0.0
        total_weight = float(sum(sentiments))
        norm = math.sqrt(len(sentiments))
        return total_weight / norm


class PartOfSentenceAnalyser(SentimentAnalyser, LemmasNormalizer):
    """
    Analyser that user WordNet corpus and word tagging procedures.
    """

    class SentimentTerm:

        def __init__(self, lemma, pos, **params):
            self.lemma = lemma
            self.pos = pos
            self.avg_positive = params.get('avg_positive', 0.0)
            self.avg_negative = params.get('avg_negative', 0.0)
            self.mean_score = params.get('mean_score', 0.0)

        @property
        def score(self):
            return self.mean_score

        def __str__(self):
            return ("Term({lemma}/{pos}, "
                    "+{avg_positive}, "
                    "-{avg_negative}, "
                    "mean={mean_score})").format(**self.__dict__)

        def __repr__(self):
            return str(self)

    def __init__(self, **params):
        super(PartOfSentenceAnalyser, self).__init__(**params)

    @classmethod
    def name(cls) -> str:
        return "SentiWordNet"

    def sentiment(self, text: str) -> float:
        sentence_scores = []
        sentences = self.normalize(text)

        for sentence in sentences:
            tagged = nltk.pos_tag(sentence)
            scored_words = [t for t in (self.score(*pair) for pair in tagged) if t]
            scores = [w.score for w in scored_words]
            sentence_sentiment_score = \
                (sum(scores) / math.sqrt(len(scores))) if scores else 0.0
            normalized = self._convert_score_to_range(sentence_sentiment_score)
            sentence_scores.append(normalized)

        if not sentence_scores:
            return 0.0

        return sum(sentence_scores) / math.sqrt(len(sentence_scores))

    def score(self, word: str, tag: str) -> SentimentTerm:
        """
        Returns score for tagged word.

        Args:
            word (str): the word to be scored
            tag (str): the word's tag
        """
        # TODO: think about better tagging
        parts = self._tag_to_wordnet_pos(tag)
        positive_count = 0
        negative_count = 0
        positive_scores = []
        negative_scores = []

        for pos in parts:

            for term in swn.senti_synsets(word, pos):
                ps = term.pos_score()
                ns = term.neg_score()

                if ps > 0:
                    positive_count += 1
                    positive_scores.append(ps)

                if ns > 0:
                    negative_count += 1
                    negative_scores.append(ns)

        if positive_count == 0 and negative_count == 0:
            return None

        avg_pos = np.mean(positive_scores) if positive_scores else 0.0
        avg_neg = np.mean(negative_scores) if negative_scores else 0.0
        n = positive_count + negative_count
        mu = (positive_count * avg_pos - negative_count * avg_neg) / n

        st = PartOfSentenceAnalyser.SentimentTerm(
            word, parts,
            avg_positive=avg_pos,
            avg_negative=avg_neg,
            mean_score=mu
        )
        return st

    @staticmethod
    def _tag_to_wordnet_pos(tag) -> tuple:
        if tag.startswith('NN'):
            return wordnet.NOUN,

        if tag.startswith('JJ'):
            return wordnet.ADJ,

        if tag.startswith('RB'):
            return wordnet.ADV,

        if tag.startswith('VB'):
            return wordnet.VERB, wordnet.ADJ_SAT

        return wordnet.NOUN,


class TwoStepAnalyser(SentimentAnalyser):
    """
    Uses two analysers to predict sentiment score. The one of analysers
    considered to be primary and its response is used in case of discrepancy
    between sentiment scores.
    """

    def __init__(self, primary: SentimentAnalyser,
                 secondary: SentimentAnalyser, **params):
        """
        Parameters:
            primary: the first analyser considered to be primary

            secondary: the second analyser used to adjust score of the first one
                when both of them agree in sentiment score sign
        """
        super(TwoStepAnalyser, self).__init__(**params)
        self.primary = primary
        self.secondary = secondary

    @classmethod
    def name(cls) -> str:
        return "TwoStep"

    def sentiment(self, text: str):
        fst = self.primary.sentiment(text)
        snd = self.secondary.sentiment(text)

        if fst == snd:
            return fst

        if np.sign(fst) == np.sign(snd):
            return (fst + snd)/2

        else:
            return fst


class WeightedAnalyser(SentimentAnalyser):
    """
    Uses a collection of analysers to calculate sentiment score as a weighted
    sum of their responses.
    """

    def __init__(self, *analysers, **params):
        super(WeightedAnalyser, self).__init__(**params)
        self.analysers = analysers

        n = len(self.analysers)
        default_weights = tuple([1.0/n for _ in self.analysers])
        weights = params.get("weights", default_weights)

        if len(weights) != n:
            err = "'priority' parameter should be a " \
                  "collection with {} elements".format(n)
            raise ValueError(err)

        self.weights = normalize(weights, norm='l1').tolist()[0]

    @classmethod
    def name(cls) -> str:
        return "Weighted"

    def sentiment(self, text: str) -> float:
        """
        Calculates weighted sentiment score using provided analysers.

        Args:
            text (str): the text to be analysed
        """
        results = [a.sentiment(text) for a in self.analysers]
        total_score = sum([score*w for score, w in zip(results, self.weights)])
        return total_score


class SenticNetAnalyser(SentimentAnalyser, LemmasNormalizer):
    """
    Sentiment analyser based on SenticNet concept data base.
    """

    def __init__(self, connector: SenticNetConnector, **params):
        super(SenticNetAnalyser, self).__init__(**params)
        self._connector = connector
        self._max_n = 3

    @classmethod
    def name(cls) -> str:
        return "SenticNet"

    def sentiment(self, text: str) -> float:
        sentences = self.normalize(text)
        scores = [self._sentence_score(s) for s in sentences]
        if not scores:
            return 0.0
        return sum(scores)/math.sqrt(len(scores))

    def _sentence_score(self, sent: str):
        connector = self._connector
        ngram_scores = []

        for n in range(1, self._max_n + 1):
            current_ngram_scores = []

            for phrase in ngrams(sent, n):
                concept = connector.concept(*phrase)
                if concept is None:
                    continue
                polarity = concept['polarity']
                current_ngram_scores.append(polarity)

            if not current_ngram_scores:
                continue

            norm = len(current_ngram_scores)
            total_score = sum(current_ngram_scores)/norm
            normalized_score = self._convert_score_to_range(total_score)
            ngram_scores.append(normalized_score)

        if not ngram_scores:
            return 0.0

        most_confident_score = max(ngram_scores, key=abs)
        return most_confident_score
