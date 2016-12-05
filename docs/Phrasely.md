Phrasely Sentiment Engine
=========================

Simple sentiment analysis engine. The main purpose of the classes from this package is
to accept a string or text paragraph (in English) and return sentiment score (positive,
neutral or negative).

The engine consists of the following packages:

+ `analysers` - contains several approaches to sentence sentiment analysis;
+ `preprocessors` - several mixins for text cleaning before actual sentiment analysis;
+ `sentic` - a wrapper over locally stored SenticNet sentiment corpus; is not used now;
+ `factory` - a couple of suitable functions to unify sentiment analysers initialization.

Each analyser utilizes same API and basically has the only one main method `sentiment(text: str)`
that accepts a single string that can be a sentence or a text section. It returns single floating
point value that reflect text's sentiment level. Also there is a `verbose(score: float)` method
that maps numeric score into more human-friendly string.

Here is an example of usage:
```python
# somehow retrieve text to be analysed
text = get_text()

# instantiate analyser with lookup files
analyser = SimpleSentimentAnalyser(words_path="words.txt", sentences_path="sentences.txt")

# get sentiment score (floating number) for text
sentiment_score = analyser.sentiment(text)

# convert floating number representing score into more human-readable form
emotion = analyser.verbose(score)
```

Default verbose values for scores are defined in `DEFAULT_SENTIMENT_LEVELS` 
dictionary and can be overridden by passing new dictionary with `sentiment_table` 
keyword argument into analyser constructor.

The `preprocessors` module contains a couple of mixin classes to be used while
creating new analyser to provide text normalization and cleaning 
functionality.

And the `factory` module provides some suitable wrappers to simplify analyser initialization
using its name or class object and default parameters. For example, `Simple` analyser
could be initialized as follows:
```Python 
# initialize analyser using its class object
from sentiment.factory import create_analyser
from sentiment.weights import SimpleSentimentAnalyser
analyser = create_analyser(SimpleSentimentAnalyser)

# or using analyser alias name
from sentiment.factory create_analyser_from_name
analyser = create_analyser_from_name('simple')
```

The sentiment analysis algorithms can be called via command line interface as follows:
```bash
> cd vector_scient
> python -m sentiment -n Weighted -s "All work and no play makes Jack a dull boy" -c
-0.675
```

The full list of arguments and usage description can be retrieved with the following command:
```bash
> python -m sentiment -h
```


> Note: the NLTK module depends on it's corpora and packages. To install 
> them on staging server or any other machine, NLTK downloader tool 
> should be used, i.e.:
> ```python
> import nltk
> nltk.download()
> ```
>
> Then the following packages need to be installed:
>   - sentiwordnet
>   - punkt
>   - pros_cons (for test suit only)
>   - product_reviews_1 (for test suit only)
>   - opinion_lexicon (for test suit only)
