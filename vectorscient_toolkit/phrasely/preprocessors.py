"""
A couple of mixins to extend sentiment analysers with text preprocessing
functionality.
"""
import nltk

from ..utils import remove_extra_symbols, clean_contractions


class TextNormalizer:

    def normalize(self, text: str):
        """
        Prepares passed text before analysis.
        """
        pass


class NullNormalizer(TextNormalizer):

    def normalize(self, text: str):
        return text


class LemmasNormalizer(TextNormalizer):

    SENT_DETECTOR = 'tokenizers/punkt/english.pickle'

    def normalize(self, text: str):
        """
        Splits text into sentences and each sentence - into list of words,
        removing extra symbols and empty words.
        """
        lower_case = text.lower().strip()
        contractions_replaced = clean_contractions(lower_case)
        sent_detector = nltk.data.load(self.SENT_DETECTOR)

        lem = nltk.WordNetLemmatizer()
        sentences = sent_detector.tokenize(contractions_replaced)
        processed = []

        for sentence in sentences:
            words = nltk.wordpunct_tokenize(sentence)
            ascii_only = [remove_extra_symbols(w) for w in words]
            non_empty = [w for w in ascii_only if w]
            lemmas = [lem.lemmatize(word) for word in non_empty]
            processed.append(lemmas)

        return processed
