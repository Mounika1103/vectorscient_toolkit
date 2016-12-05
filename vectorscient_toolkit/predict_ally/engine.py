"""
The PredictAlly engine entry point.
"""
import logging

from ..exceptions.exc import WrongDataTypeError
from ..sources.factory import create_data_source_from_config
from ..sources.data import DataSource
from .mixins import ConfigParsingMixin


log = logging.getLogger(__name__)
log.setLevel(level=logging.DEBUG)


class PredictAlly(ConfigParsingMixin):

    def __init__(self, user, **engine_config):
        self._techniques = []
        self._sources = []
        self._user = user
        self._user_config = {}
        self._prepare_sources = engine_config.get("prepare_sources", True)
        self._save_report = engine_config.get("save_report", True)

    @property
    def user_config(self):
        if self._user is None:
            return None
        if not self._user_config:
            self._user_config = self.parse_config(self._user.parameters)
        return self._user_config

    @property
    def techniques(self):
        """
        Returns:
            List[PredictAllyTechnique]: A list of algorithms to be applied.
        """
        return self._techniques

    @property
    def sources(self):
        """
        Returns:
            List[DataSource]: A list of data sources to be processed with the
                algorithms.
        """
        return self._sources

    def attach_technique(self, technique):
        self._techniques.append(technique)

    def load_data(self, source):
        if not isinstance(source, DataSource):
            err = "Unexpected data source type: '{}'".format(type(source))
            log.error(err)
            raise WrongDataTypeError(err)
        if not source.ready and self._prepare_sources:
            source.prepare()
        self._sources.append(source)

    def load_data_from_config(self):
        config = self.user_config
        source = create_data_source_from_config(config.records)
        self.load_data(source)

    def run(self):
        user = self._user
        for tech in self.techniques:
            for source in self.sources:
                tech.configure(**user.parameters)
                tech.run(source)

    def report(self, context):
        config = self.user_config
        self._user_config = config
        for tech in self.techniques:
            tech.report(context)
        if self._save_report and config:
            context.save(**config.reporting)
