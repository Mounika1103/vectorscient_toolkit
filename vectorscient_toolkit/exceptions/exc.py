"""
Custom exception classes for developed modules.
"""


class ApplicationException(Exception):
    pass


class IpInfoError(ApplicationException):
    pass


class DataSourceError(ApplicationException):
    """
    Thrown if error occurred during data source initialization or preparation.
    """


class StatsError(ApplicationException):
    pass


class ConfigurationError(ApplicationException):
    """
    Thrown if bad configuration file was read or incorrect values were found.
    """


class InvalidDataError(ApplicationException):
    """
    Thrown if bad data received or data became malformed after processing.
    """


class ReportingContextError(ApplicationException):
    """
    Thrown if reporting context improperly configured or loaded with
    wrong data format.
    """


class SentimentAnalysisError(ApplicationException):
    """
    Thrown if sentiment analysis goes wrong.
    """


class ClusteringPipelineError(ApplicationException):
    """
    Thrown if clustering pipeline unexpectedly fails on some step.
    """


class PredictAllyError(ApplicationException):
    """
    Common base class for exceptions thrown by PredictAlly.
    """


class WrongDataTypeError(PredictAllyError):
    """
    Thrown if unexpected data type was passed.
    """


class DataProcessingError(PredictAllyError):
    """
    Thrown if data preprocessing pipeline improperly configured.
    """
