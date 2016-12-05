from collections import ChainMap
from itertools import chain

from reportlab.platypus import BaseDocTemplate

from .entry import ReportEntry, VectorScientHeadlineManager


class PDFReport:
    """
    Generates a data analysis report using provided primitives.
    """

    def __init__(self, **report_config):
        """
        Args:
            pagesize:
            **report_config:
        """
        self._report_entries = []
        self._report_config = report_config
        self._title = report_config.get("title", "Data Analysis Report")
        self._disclaimer = report_config.get("disclaimer", "")
        self._doc = None

    @property
    def parameters(self):
        return self._report_config

    def add_entry_to_report(self, entry: ReportEntry):
        self._report_entries.append(entry)

    def _generate(self, doc):
        contents = (e.get_content(doc) for e in self._report_entries)
        non_emtpy_contents = [c if isinstance(c, list) else [c]
                              for c in contents if c is not None]
        return list(chain(*non_emtpy_contents))

    def save(self, file_name, **pdf_params):
        joined_params = dict(ChainMap(pdf_params, self._report_config))
        doc = BaseDocTemplate(file_name, **joined_params)
        template_elements = self._generate(doc)
        manager = VectorScientHeadlineManager(disclaimer=self._disclaimer)
        manager.insert_into_report(doc)
        doc.build(template_elements)
        self._doc = doc
