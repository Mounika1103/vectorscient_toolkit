from unittest import mock
import os

from faker import Factory
import pandas as pd
import numpy as np

from ..predict_ally.shortcuts import (
    run_opportunity_clustering_for_registered_user,
    run_opportunity_clustering_for_trial_user
)
from ..utils import data_path


def create_fake_data_of_size(data_size):
    fake = Factory.create()
    data = pd.DataFrame({
        "num1": np.random.randn(data_size),
        "num2": np.random.randn(data_size),
        "num3": np.random.randn(data_size),
        "const": np.zeros(data_size),
        "attr1": [fake.name() for _ in range(data_size)],
        "attr2": [fake.ipv4() for _ in range(data_size)]
    })
    return data


class TestTrialUserShortcuts:

    def setup(self):
        self.data = create_fake_data_of_size(100)

    def test_running_opportunity_clustering_with_PDF_context(self):
        pandas_path = 'vectorscient_toolkit.sources.data.pd.read_csv'

        # use PDF context to save data
        with mock.patch(pandas_path) as mock_read_csv:
            mock_read_csv.return_value = self.data
            path_to_csv = "/path/to/csv"
            pdf_name = "trial_user_report.pdf"

            run_opportunity_clustering_for_trial_user(
                path_to_csv, {
                    'reporting': {
                        'context': ['pdf'],
                        'file_name': pdf_name
                    }
                })

            mock_read_csv.assert_called_once_with(path_to_csv)
            assert os.path.exists(pdf_name)

    def test_running_opportunity_clustering_with_archive_context(self):
        pandas_path, context_save_path = (
            'vectorscient_toolkit.sources.data.pd.read_csv',
            'vectorscient_toolkit.predict_ally.shortcuts.ArchiveContext.save'
        )

        # use archive context to save data
        with mock.patch(pandas_path) as mock_read_csv:
            mock_read_csv.return_value = self.data

            with mock.patch(context_save_path) as mock_save:
                path_to_csv = "/path/to/csv"
                archive_name = "arch"

                run_opportunity_clustering_for_trial_user(path_to_csv, {
                    'reporting': {'file_name': archive_name}
                })

                assert mock_read_csv.called
                mock_read_csv.assert_called_once_with(path_to_csv)
                mock_save.assert_called_once_with(file_name="arch")

    def test_running_opportunity_clustering_with_several_contexts(self):
        overridden_config = {'reporting': {'context': ['pdf', 'archive']}}
        pandas_path, archive_context_path, pdf_context_path = (
            'vectorscient_toolkit.sources.data.pd.read_csv',
            'vectorscient_toolkit.predict_ally.shortcuts.ArchiveContext.save',
            'vectorscient_toolkit.predict_ally.shortcuts.PDFContext.save'
        )

        with mock.patch(pandas_path) as mock_read_csv:
            mock_read_csv.return_value = self.data

            with mock.patch(archive_context_path) as mock_archive_ctx:
                with mock.patch(pdf_context_path) as mock_pdf_ctx:
                    path_to_csv = "path_to_csv"

                    run_opportunity_clustering_for_trial_user(
                        path_to_csv, overridden_config)

                    assert mock_archive_ctx.call_count == 1
                    assert mock_pdf_ctx.call_count == 1


class TestRegisteredUserShortcuts:

    def test_running_opprotunity_clustering_with_PDF_context(self):
        pdf_name = "registred_user_report.pdf"
        config_path = data_path('settings/config.remote.v2.yaml')
        run_opportunity_clustering_for_registered_user(
            config_path, output_file=pdf_name, context='pdf')

    def test_running_opportunity_clustering_with_archive_context(self):
        local_path = data_path('settings/config.local.v2.yaml')
        run_opportunity_clustering_for_registered_user(local_path, context='archive')

        remote_path = data_path('settings/config.remote.v2.yaml')
        run_opportunity_clustering_for_registered_user(remote_path, context='archive')

    def test_running_opportunity_clustering_with_several_contexts(self):
        local_path = data_path('settings/config.remote.v2.yaml')
        contexts = ['pdf', 'archive']
        run_opportunity_clustering_for_registered_user(
            local_path, context=contexts)
