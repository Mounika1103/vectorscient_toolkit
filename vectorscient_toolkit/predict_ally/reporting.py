"""
Possible reporting contexts for prediction algorithms.

Each algorithm can accept a ReportingContext subclass instance to output
prediction results in suitable form, for example, as an archive with images,
as a single image, PDF file, etc.
"""
import zipfile
import uuid
import abc
import io
import os

import pandas as pd

from ..pdf.entry import PlainTextEntry, TextualTableEntry, BreakEntry, PageHeader
from ..pdf.entry import ListOfItemsEntry, ImageEntry
from ..exceptions.exc import ReportingContextError
from ..pdf.report import PDFReport


class ReportingContext(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def add_to_report(self, data):
        pass

    def save(self, **params):
        """
        Saves prepared report, i.e. writes into filesystem, prints into
        standard output, etc. Default implementation does nothing.
        """
        pass


class ImageFileContext(ReportingContext):
    """
    Saves data processing results into image.
    """

    def __init__(self, **params):
        self._data = io.BytesIO()

    def add_to_report(self, data: io.BytesIO):
        data.seek(0)
        self._data.write(data.getbuffer())

    def save(self, **params):
        self._data.seek(0)
        image_path = params.get("file_name", str(uuid.uuid4()) + ".png")
        with open(image_path, "wb") as img:
            img.write(self._data.getbuffer())


class ArchiveContext(ReportingContext):
    """
    Packs data processing results into archive.
    """

    def __init__(self, **params):
        self._entries = {}
        self._saved_to = None

    def add_to_report(self, archive_entry):
        name, item = archive_entry

        if isinstance(item, str):
            data = item.encode()

        elif isinstance(item, io.StringIO):
            item.seek(0)
            data = item.getvalue().encode()

        elif isinstance(item, bytes):
            data = item

        elif isinstance(item, io.BytesIO):
            item.seek(0)
            data = item.getbuffer()

        elif isinstance(item, pd.DataFrame):
            buf = io.StringIO()
            item.to_csv(buf)
            buf.seek(0)
            data = buf.getvalue().encode()

        else:
            err = "Cannot add to archive entry of type: '{}'".format(type(item))
            raise ReportingContextError(err)

        self._entries[name] = data

    def save(self, **params):
        archive_name = params.get("file_name", str(uuid.uuid4()))
        archive_type = params.get("type", "zip")
        file_name = "{}.{}".format(archive_name, archive_type)
        with zipfile.ZipFile(file_name, mode="w") as arch:
            for name_in_archive, data in self._entries.items():
                arch.writestr(name_in_archive, data)
        self._saved_to = file_name


class PDFContext(ReportingContext):
    """
    Saves data processing results into PDF document.
    """

    def __init__(self, report_builder=None, **params):
        if report_builder is None:
            self._builder = PDFReport(**params)
        else:
            self._builder = report_builder
        self._saved_to = None

    def add_to_report(self, data, **item_params):
        self._add_item(data, **item_params)

    def break_page(self):
        self._builder.add_entry_to_report(BreakEntry(mode="pagebreak"))

    def break_page_with_header(self, header: str):
        self.break_page()
        self._builder.add_entry_to_report(PageHeader(header))

    def _add_item(self, item, **item_params):
        report = self._builder

        if isinstance(item, str):
            report.add_entry_to_report(PlainTextEntry(text=item, **item_params))

        elif isinstance(item, pd.Series):
            header = item.name
            list_of_items = item
            entry = ListOfItemsEntry(list_of_items, header)
            report.add_entry_to_report(entry)

        elif isinstance(item, pd.DataFrame):
            report.add_entry_to_report(
                TextualTableEntry(item, **item_params))

        elif isinstance(item, (bytes, io.BytesIO)):
            report.add_entry_to_report(ImageEntry(item, width=450))

        else:
            err = "Cannot add to PDF entry of type: '{}'".format(type(item))
            raise ReportingContextError(err)

    def save(self, **params):
        file_name = params.get("file_name", str(uuid.uuid4()))
        name, _ = os.path.splitext(file_name)
        cleaned_file_name = name + ".pdf"
        self._builder.save(cleaned_file_name)
        self._saved_to = cleaned_file_name
