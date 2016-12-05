"""
Suitable wrapper to unify sentiment analysers initialization using default
parameters.
"""
from .sentic import SenticNetLocalStorage
from .analysers import (
    SentimentAnalyser,
    SimpleSentimentAnalyser,
    PartOfSentenceAnalyser,
    SenticNetAnalyser,
    WeightedAnalyser,
    TwoStepAnalyser,
)

from ..config import DEFAULT_ANALYSER
from ..utils import data_path


def enumerate_analysers():
    yield SimpleSentimentAnalyser
    yield PartOfSentenceAnalyser
    yield SenticNetAnalyser
    yield WeightedAnalyser
    yield TwoStepAnalyser


def create_analyser(cls) -> SentimentAnalyser:
    """
    Creates an instance of specified SentimentAnalyser successor using default
    parameters and data paths.
    """
    if cls is SimpleSentimentAnalyser:
        words_path = data_path('sentiment_weights/words.txt')
        sents_path = data_path('sentiment_weights/sentences.txt')
        analyser = SimpleSentimentAnalyser(
            words_path=words_path, sentences_path=sents_path)
        return analyser

    elif cls is PartOfSentenceAnalyser:
        return PartOfSentenceAnalyser()

    elif cls is TwoStepAnalyser:
        fst = create_analyser(SimpleSentimentAnalyser)
        snd = create_analyser(PartOfSentenceAnalyser)
        return TwoStepAnalyser(fst, snd)

    elif cls is WeightedAnalyser:
        fst = create_analyser(SimpleSentimentAnalyser)
        snd = create_analyser(PartOfSentenceAnalyser)
        return WeightedAnalyser(fst, snd, weights=[0.7, 0.3])

    elif cls is SenticNetAnalyser:
        # TODO: add ZIP-archive support
        conn = SenticNetLocalStorage(data_path('sentic_net'))
        return SenticNetAnalyser(connector=conn)

    else:
        raise ValueError("Unexpected analyser type: {}".format(str(cls)))


def create_analyser_from_name(name: str) -> SentimentAnalyser:
    """
    Creates a sentiment analyser instance using its verbose name.
    """
    for cls in enumerate_analysers():
        if cls.name() == name:
            return create_analyser(cls)

    available = "\n".join(["\t- " + a.name() for a in enumerate_analysers()])
    err = "Unexpected analyser name '{}'. It should be one the following:\n{}"
    raise ValueError(err.format(name, available))


default_analyser = create_analyser_from_name(DEFAULT_ANALYSER)
