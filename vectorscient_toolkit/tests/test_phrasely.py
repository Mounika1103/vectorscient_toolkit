from collections import namedtuple
import os

import numpy as np
import prettytable
import pytest
import nltk

from vectorscient_toolkit.phrasely.analysers import DEFAULT_SENTIMENT_LEVELS
from vectorscient_toolkit.phrasely.analysers import SimpleSentimentAnalyser
from vectorscient_toolkit.phrasely.analysers import PartOfSentenceAnalyser
from vectorscient_toolkit.phrasely.analysers import SenticNetAnalyser
from vectorscient_toolkit.phrasely.factory import enumerate_analysers
from vectorscient_toolkit.phrasely.factory import create_analyser


@pytest.fixture
def texts():
    sentences = [
        "Finn is stupid and idiotic",
        "Finn is only a tiny bit stupid and not idiotic",
        "Nothing is impossible",
        "I couldn't agree more with you",
        "It is not what you may expect",
        "Rubbish",
        "This item cannot be any better",
        "This item can't be any better",
        "I like this movie so much!",
        "I love ham",
        "I hate spam",
        "Iain said it was amazing from onstage",
        "The service could be much better than it is",
        "Amazing places where your dollar goes further this year",
        "When the movie boring during netflix and chill",
        "I trust no one on April Fools. I hate this 'holiday'",
        "The most ASTONISHING and shaming sentence in the whole contract",
        "We need to promote love, unity, & the care for our planet & future generations",
        "Home of the Stanley Hotel. All work and no play makes Marko a dull boy",
        "You do have a pretty interesting life"
    ]

    long_random_text = (
        "Ask especially collecting terminated may son expression. "
        "Extremely eagerness principle estimable own was man. Men "
        "received far his dashwood subjects new. My sufficient "
        "surrounded an companions dispatched in on. Connection too "
        "unaffected expression led son possession. New smiling friends "
        "and her another. Leaf she does none love high yet. Snug love "
        "will up bore as be. Pursuit man son musical general pointed. "
        "It surprise informed mr advanced do outweigh.")

    min_score = min(DEFAULT_SENTIMENT_LEVELS.keys())
    max_score = max(DEFAULT_SENTIMENT_LEVELS.keys())

    old_path = nltk.data.path
    nltk.data.path.append(os.environ['NLTK_DATA_PATH'])
    yield sentences, long_random_text, min_score, max_score
    nltk.data.path = old_path


class TestPhrasely:

    def test_simple_analyser_score_calculation(self, texts):
        self.validate_analyser(create_analyser(SimpleSentimentAnalyser), texts)

    def test_senti_word_net_analyser_score_calculation(self, texts):
        self.validate_analyser(create_analyser(PartOfSentenceAnalyser), texts)

    def test_sentic_net_analyser_score_calculation(self, texts):
        self.validate_analyser(create_analyser(SenticNetAnalyser), texts)

    def test_analysers_scoring_consistence(self, texts):
        analysers = tuple(create_analyser(a) for a in enumerate_analysers())
        headers = ['#', 'Sentence', 'Mismatch'] + [a.name() for a in analysers]
        table = prettytable.PrettyTable(field_names=headers)
        sentences, _, _, _ = texts

        for i, s in enumerate(sentences):
            scores = [analyser.sentiment(s) for analyser in analysers]
            same_sign = len(set(map(np.sign, scores))) == 1
            formatted = ["{:2.2f}".format(s) for s in scores]

            row = [i, s, ' ' if same_sign else 'x']
            row.extend(formatted)
            table.add_row(row)

        table.align = 'r'
        table.align["#"] = 'c'
        table.align["Sentence"] = 'l'
        table.align["Mismatch"] = 'c'

        print("\nAnalysers comparison table\n")
        print(table)

    def validate_analyser(self, analyser, texts):
        sentences, long_random_text, min_score, max_score = texts
        scores = [analyser.sentiment(s) for s in sentences]
        single_score = analyser.sentiment(long_random_text)

        assert isinstance(scores, (list, set, tuple))
        assert isinstance(single_score, float)
        assert all(min_score <= s <= max_score for s in scores)

