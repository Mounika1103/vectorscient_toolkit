from unittest import mock

import pandas as pd

from ..predict_ally.clustering.unsupervised import OpportunityClassifier
from ..predict_ally.engine import PredictAlly
from ..predict_ally.reporting import ArchiveContext
from ..predict_ally.technique import PredictAllyTechnique
from ..sources.data import DataSource
from ..utils import data_path


class TestEngine:

    def test_engine_launching(self):
        # retrieve user parameters
        user = mock.MagicMock(account='registered',
                              parameters={'config_file': 'path/to/config.yaml'})

        # initialize engine with user's configuration
        engine = PredictAlly(user)

        # choose a technique to apply
        technique = mock.create_autospec(PredictAllyTechnique)
        engine.attach_technique(technique)

        assert len(engine.techniques) == 1, (
            "The data processing algorithm wasn't attached")

        # create a data source
        source = mock.create_autospec(DataSource)
        source.ready = False
        engine.load_data(source)

        source.prepare.assert_any_call()
        assert len(engine.sources) == 1, "The data source wasn't attached"

        # run selected algorithm on loaded data
        engine.run()

        technique.configure.assert_called_once_with(**user.parameters)
        technique.run.assert_called_once_with(source)

    def test_unsupervised_clustering_with_local_CSV_data_source(self):
        # consider a registered user
        user = mock.MagicMock(
            account='registered',
            parameters={'config_file': data_path('settings/config.local.v2.yaml')})

        # initialize engine for that user
        engine = PredictAlly(user)
        engine.load_data_from_config()

        assert len(engine.sources) == 1
        assert engine.sources[0].ready

        # initialize clustering algorithm without user's config
        clustering = OpportunityClassifier()
        engine.attach_technique(clustering)

        # run algorithm with data
        engine.run()
        results = clustering.model()

        assert isinstance(results.clusters, pd.DataFrame)
        assert not results.clusters.empty

        # report results
        context = ArchiveContext()
        engine.report(context)
