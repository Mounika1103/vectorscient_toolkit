from unittest import mock
import tempfile
import io

import pandas as pd
import pytest

from ..predict_ally.mixins import DataPreprocessingMixin
from ..predict_ally.reporting import ImageFileContext
from ..predict_ally.mixins import ConfigParsingMixin
from ..exceptions.exc import DataProcessingError
from ..exceptions.exc import ConfigurationError
from ..predict_ally.reporting import PDFContext
from ..pdf.report import PDFReport


class TestConfigParsingMixin:

    def test_parsing_config_from_dict(self):
        config = {"first": 1, "second": 2, "third": 3}

        mixin = ConfigParsingMixin()
        parsed = mixin.parse_config(config)
        assert len(parsed) == len(config)

        for key in config.keys():
            assert key in parsed
            assert parsed[key] is not None
            assert config[key] == parsed[key]

    def test_parsing_config_from_yaml(self):
        file_name = self.prepare_yaml_config()

        mixin = ConfigParsingMixin()
        parsed = mixin.parse_config(file_name)

        assert "first_key" in parsed
        assert "second_key" in parsed
        assert "third_key" in parsed
        assert parsed["third_key"] == ["one", "two"]

    def test_throw_exception_on_wrong_config_format(self):
        mixin = ConfigParsingMixin()

        config_list = ["first", 1, "second", 2]

        with pytest.raises(ConfigurationError):
            mixin.parse_config(config_list)

        config_file = "config.ini"
        with pytest.raises(ConfigurationError):
            mixin.parse_config(config_file)

    @staticmethod
    def prepare_yaml_config():
        config = "\n".join(["first_key: 1",
                            "second_key: Yes",
                            "third_key:",
                            "  - one",
                            "  - two"])
        config_file = tempfile.NamedTemporaryFile(
            suffix='.yaml', mode='w', delete=False)
        config_file.write(config)
        file_name = config_file.name
        config_file.close()
        return file_name


class TestDataPreprocessingMixin:

    def test_registering_new_preprocessor_with_specific_order(self):
        mixin = DataPreprocessingMixin()
        mixin.register_preprocessor(id, name="First", order=2)
        mixin.register_preprocessor(id, name="Second", order=1)

        first = mixin.pipeline[1]
        second = mixin.pipeline[2]

        assert len(mixin.pipeline) > 0
        assert first["name"] == "Second"
        assert second["name"] == "First"

    def test_data_preprocessing_in_accordance_with_specified_order(self):
        mixin = DataPreprocessingMixin()
        self.register_plus_one_preprocessor(mixin)
        self.register_times_two_preprocessor(mixin)
        data = pd.DataFrame({"x": [1, 2, 3]})

        result = mixin.apply_preprocessing(data, keep_intermediate=True)

        assert list(result["x"]) == [3, 5, 7]

    def test_throw_exception_on_badly_configured_pipeline(self):
        mixin = DataPreprocessingMixin()
        mixin.register_preprocessor(lambda x: None, name="BadMapping")
        data = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(DataProcessingError):
            mixin.apply_preprocessing(data)

    @staticmethod
    def register_plus_one_preprocessor(mixin):
        def plus_one(x):
            return x + 1
        mixin.register_preprocessor(plus_one, name='PlusOne', order=10)

    @staticmethod
    def register_times_two_preprocessor(mixin):
        def times_two(x):
            return x * 2
        mixin.register_preprocessor(times_two, name='TimesTwo', order=5)


class TestReportingContexts:

    @mock.patch('vectorscient_toolkit.predict_ally.reporting.open', create=True)
    def test_image_file_reporting_context(self, mock_open):
        mock_open.return_value = mock.MagicMock(spec=io.IOBase)
        mock_image = mock.create_autospec(io.BytesIO)
        data = b'image data'
        mock_image.getbuffer.return_value = data

        context = ImageFileContext()
        context.add_to_report(mock_image)
        context.save()

        assert mock_image.getbuffer.called
        mock_image.seek.assert_called_once_with(0)
        file_handle = mock_open.return_value.__enter__.return_value
        file_handle.write.assert_called_once_with(data)

    def test_archive_reporting_context(self):
        pass

    def test_PDF_reporting_context(self):
        string = 'plain text string'
        image = b'image data'
        table = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        mock_report = mock.create_autospec(PDFReport)

        context = PDFContext(report_builder=mock_report)
        context.add_to_report(string)
        context.add_to_report(image)
        context.add_to_report(table)
        context.save(file_name='report.pdf')

        mock_report.save.assert_called_once_with(file_name='report.pdf')
        assert mock_report.add_entry_to_report.call_count == 3
