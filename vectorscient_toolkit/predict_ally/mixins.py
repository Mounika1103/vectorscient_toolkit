"""
Mixins to provide access for commonly demanded functionality and utils.
"""
from os import path
import logging
import json

import attrdict
import yaml

from ..exceptions.exc import ConfigurationError, DataProcessingError


log = logging.getLogger(__name__)
log.setLevel(level=logging.DEBUG)


class ConfigParsingMixin:
    """
    Implements common config parsing functionality used by all PredictAlly
    techniques.
    """

    @staticmethod
    def parse_config(config):
        """
        Parses configuration file.

        Args:
            config (dict or str): A configuration object.

        Returns:
             config (attrdict.AttrDict): A parsed configuration
                dict-like object.
        """
        if isinstance(config, dict) and "config_file" not in config:
            return attrdict.AttrDict(config)

        elif isinstance(config, str):
            file_name = config

        elif "config_file" in config:
            file_name = config["config_file"]

        else:
            err = "Unexpected configuration object type: " + str(type(config))
            raise ConfigurationError(err)

        name, ext = path.splitext(file_name)
        ext = ext.replace(".", "")
        parsers = {"json": json, "yaml": yaml}

        if ext not in parsers:
            err = ("Configuration file has unexpected "
                   "extension: {}. Available choices "
                   "are ({})".format(ext, ", ".join(parsers.keys())))
            raise ConfigurationError(err)

        if not path.exists(file_name):
            raise ConfigurationError(
                "Config file doesn't exist: '{}'".format(file_name))

        with open(file_name) as fp:
            content = parsers[ext].load(fp)

        return attrdict.AttrDict(content)


class DataPreprocessingMixin:

    def __init__(self):
        self._intermediate_results = {}
        self._preprocessors = {}
        self._final_result = None

    @property
    def pipeline(self):
        return self._preprocessors

    def register_preprocessor(self, prep, order=None, name=None):
        """
        Adds another one preprocessor into pipeline.

        Args:
            prep: (callable): A preprocessing algorithm.
            order (int): An invocation order.
            name (str): A verbose name.
        """
        if order is None:
            already_added = list(self._preprocessors.keys())
            order = (max(already_added) + 1) if already_added else 1
        if name is None:
            n = len(self._preprocessors)
            name = "Preprocessor #" + str(n + 1)
        self._preprocessors[order] = {"name": name, "prep": prep}

    def purge_preprocessors(self):
        self._preprocessors = {}

    def apply_preprocessing(self, data, keep_intermediate=False):
        """
        Args:
            data (pandas.DataFrame):
            keep_intermediate (bool):
        """
        ordering = [order for order in sorted(self._preprocessors.keys())]
        current_data = data.copy()

        for number in ordering:
            name = self._preprocessors[number]["name"]
            prep = self._preprocessors[number]["prep"]

            log.debug("Applying data processor: '%s'", name)
            processed = prep(current_data)

            if processed is None or processed.empty:
                err = "Data processing pipeline is broken: " \
                      "None or empty data was found"
                log.error(err)
                raise DataProcessingError(err)

            if keep_intermediate:
                self._intermediate_results[name] = processed.copy()

            log.debug("Data shape after processing: %s", str(processed.shape))
            current_data = processed

        self._final_result = current_data
        return self._final_result
