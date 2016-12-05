import abc

from ..sources.data import DataSource


class DataPreprocessor(metaclass=abc.ABCMeta):

    def __call__(self, data, *args, **kwargs):
        return self.apply(data, *args, **kwargs)

    @abc.abstractclassmethod
    def apply(self, data, *args, **kwargs):
        """
        Implements specific data transformation algorithm (i.e. normalization,
        filling missed values, etc.)

        Args:
            data (pandas.DataFrame): A data frame to be preprocessed.

        Returns:
            new (pandas.DataFrame): A resultant data frame.
        """


class PredictAllyTechnique(metaclass=abc.ABCMeta):
    """
    Common base class for PredictAlly ML engine techniques.
    """

    @abc.abstractmethod
    def run(self, source):
        """
        Runs configured algorithm.

        Args:
            source (pandas.DataFrame or DataSource): A dataset to be processed.
        """

    @abc.abstractmethod
    def report(self, context):
        """
        Generates human-readable report about processed data.
        """

    @abc.abstractmethod
    def model(self) -> dict:
        """
        Returns the result of data processing with specific technique.
        """

    def benchmark(self):
        """
        Optional method to benchmark implemented algorithm.
        """
        pass

    def configure(self, **params):
        """
        Optional method to (re)configure algorithm.
        """
        pass
