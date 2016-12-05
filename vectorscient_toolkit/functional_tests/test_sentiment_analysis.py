from string import ascii_lowercase
import random
import nltk
import os

from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

from ..phrasely.factory import *
from ..utils import data_path


class TestSentimentAnalysisAccuracy:

    def setup(self):
        self.old_nltk_path = nltk.data.path[:]
        os.environ['VECTORSCIENT_DATA_PATH'] = \
            os.path.expanduser('~/Code/Tasks/vectorscient')
        nltk.data.path.append(os.environ['NLTK_DATA_PATH'])

        self.analysers = [
            create_analyser_from_name('Simple'),
            create_analyser_from_name('SentiWordNet'),
            create_analyser_from_name('Weighted')
        ]

    def teardown(self):
        del os.environ['VECTORSCIENT_DATA_PATH']
        nltk.data.path = self.old_nltk_path

    def test_scoring_on_labelled_dataset(self):
        # randomly choose labelled sentences
        labelled_sample = self.sample_data_from_pros_cons_corpus(500)

        # use available analysers
        analysers = self.analysers
        scores = []

        for sent, label in labelled_sample:
            result = {a.name(): a.sentiment(sent) for a in analysers}
            result.update({'Sentence': sent, 'Label': label})
            scores.append(result)

        df = pd.DataFrame(scores)
        df = df[['Sentence', 'Label'] + [a.name() for a in analysers]]
        df.to_csv(data_path('sentiment_scores.csv'), index=None)

    def test_scoring_accuracy(self):
        analysers = self.analysers
        data_sample = self.sample_data_from_pros_cons_corpus(all_data=True)

        scores = []
        for sent, label in data_sample:
            result = {a.name(): a.sentiment(sent) for a in analysers}
            result.update({'Sentence': sent, 'Label': label})
            scores.append(result)

        df = pd.DataFrame(scores)
        labels = df['Label']
        conf_matrices = []

        for a in analysers:
            preds = [(+1 if p >= 0 else -1) for p in df[a.name()]]
            tp, fn, fp, tn = np.ravel(confusion_matrix(labels, preds))
            m = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'analyser': a.name()}
            conf_matrices.append(m)

        results = pd.DataFrame(conf_matrices)
        results.to_csv(data_path('conf_matrices.csv'))

    def test_scoring_separate_opinion_words(self):
        analysers = self.analysers

        # choose a set of lexicon words
        labelled_data = self.sample_data_from_opinion_lexicon_corpus(size=1000)
        words = [word for (word, score) in labelled_data]
        scores = []

        for word in words:
            result = {a.name(): a.sentiment(word) for a in analysers}
            if not any([abs(v) > 0.5 for v in result.values()]):
                continue
            result['Word'] = word
            scores.append(result)

        df = pd.DataFrame(scores)
        df.to_csv(data_path('lexicon.csv'), index=False)

    @staticmethod
    def sample_data_from_sentence_polarity_corpus(size=100):
        from nltk.corpus import sentence_polarity
        neg_file, pos_file = sentence_polarity.fileids()
        negative_sentences = sentence_polarity.sents(neg_file)
        positive_sentences = sentence_polarity.sents(pos_file)
        neg = [(s, -1) for s in negative_sentences]
        pos = [(s, +1) for s in positive_sentences]
        all_sentences = neg + pos
        random.shuffle(all_sentences)
        return random.sample(all_sentences, size)

    @staticmethod
    def sample_data_from_reviews_corpus(size=100):
        from nltk.corpus import product_reviews_1 as reviews_corp

        reviews = list(reviews_corp.reviews(reviews_corp.fileids()))
        random.shuffle(reviews)
        sample = random.sample(reviews, 100)
        allowed = set(ascii_lowercase + '-')
        testing_set = []

        for review in sample:
            scores = [
                int(score)
                for rl in review.review_lines
                for (feat, score) in rl.features]

            if not scores:
                continue

            raw_sents = [
                ' '.join([w for w in rl.sent if set(w).issubset(allowed)])
                for rl in review.review_lines]

            sents = [
                (sent.replace("do n t", "don't").replace("didn t", "didn't")
                 .replace("don t", "don't").replace("did n t", "didn't")
                 .replace("can t", "cannot").replace("can't", "cannot"))
                for sent in raw_sents if sent]

            avg_score = np.mean(scores)
            entry = (". ".join(sents), "+" if avg_score > 0 else "-")
            testing_set.append(entry)

        random.shuffle(testing_set)
        return random.sample(testing_set, size)

    @staticmethod
    def sample_data_from_pros_cons_corpus(size=100, all_data=False):
        from nltk.corpus import pros_cons
        from html import unescape
        import random
        cons_id, pros_id = pros_cons.fileids()
        cons_lines = [
            (unescape(con.replace('<Cons>', '').replace('</Cons>', '')).strip(), -1)
            for con in pros_cons.open(cons_id) if con]
        pros_lines = [
            (unescape(pro.replace('<Pros>', '').replace('</Pros>', '')).strip(), +1)
            for pro in pros_cons.open(pros_id) if pro]

        verbose_cons = [(c, l) for (c, l) in cons_lines if len(c.split()) > 3]
        verbose_pros = [(p, l) for (p, l) in pros_lines if len(p.split()) > 3]

        all_lines = verbose_pros + verbose_cons

        if not all_data:
            shortened = [line[:100] for line in all_lines]
            random.shuffle(shortened)
            return random.sample(shortened, size)

        else:
            return all_lines

    @staticmethod
    def sample_data_from_opinion_lexicon_corpus(size=100):
        from nltk.corpus import opinion_lexicon
        positives = [(w, +1) for w in opinion_lexicon.positive()]
        negatives = [(w, -1) for w in opinion_lexicon.negative()]
        sample = random.sample(positives, size) + random.sample(negatives, size)
        random.shuffle(sample)
        return sample
