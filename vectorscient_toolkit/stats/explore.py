import os
import io

from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus.flowables import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import matplotlib.pyplot as plt
import pandas as pd

from ..sources.data import CsvDataSource


class DataExplorationReport:

    def __init__(self, output_file: str, **params):
        self._entries = []
        self._template_elements = []
        self._output_file = output_file
        self._template = SimpleDocTemplate(self._output_file, pagesize=A4)
        self._title = params.get("title", "Distribution Analysis Report")

    def add_element(self, entry):
        if not isinstance(entry, ReportEntry):
            raise TypeError("'ReportEntry' instance expected, "
                            "got '{}'".format(type(entry)))
        self._entries.append(entry)

    def generate(self):
        for entry in self._entries:
            entry.append_to_report(self)
        self.save()

    def save(self):
        self._template.build(
            self._template_elements,
            onFirstPage=lambda c, d: self._create_report_title(c, d))

    def append_table(self, headers: list, content: list, style: TableStyle):
        """
        Appends table to report template.

        Args:
            headers (list): an array of table headers
            content (list): a matrix with table content
            style (TableStyle): custom style to be applied
        """
        whole_table = [headers] + content
        table = Table(whole_table)
        table.setStyle(style)
        self._template_elements.append(table)

    def append_image(self, image_or_bytes):
        if isinstance(image_or_bytes, bytes):
            img = Image(image_or_bytes)
        else:
            img = image_or_bytes
        self._template_elements.append(img)

    def append_paragraph(self, text, style):
        p = Paragraph(text, style)
        self._template_elements.append(p)
        self._template_elements.append(Spacer(0, 10))

    def _create_report_title(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 16)
        width, height = A4
        canvas.drawCentredString(width/2.0, height - 54, self._title)
        canvas.restoreState()


class ReportEntry:

    def __init__(self, **params):
        self._index = params.get('index', None)

    def append_to_report(self, report):
        pass


class PlainTextEntry(ReportEntry):

    def __init__(self, **params):
        super(PlainTextEntry, self).__init__(**params)
        self._text = params.get("text", "")

    def append_to_report(self, report):
        if not self._text:
            return

        styles = getSampleStyleSheet()
        style = styles["Normal"]

        report.append_paragraph(self._text, style)


class TextualTableEntry(ReportEntry):

    def __init__(self, **params):
        super(TextualTableEntry, self).__init__(**params)
        self._table = params.get('table', pd.DataFrame())
        self._title = params.get('header', '')
        self._subtitle = params.get('subtitle', '')

    def append_to_report(self, report):
        if self._table.empty:
            return

        def to_fixed(row):
            updated = []
            for item in row:
                if isinstance(item, float):
                    item = str(round(item, ndigits=4))
                updated.append(item)
            new = pd.Series(updated)
            new.name = row.name
            return new

        df = self._table
        rounded = df.apply(to_fixed, axis=1)
        rounded.columns = df.columns

        header = rounded.columns.tolist()
        content = rounded.values.tolist()

        style = TableStyle()
        border_width = 0.25
        style.add('INNERGRID', (0, 0), (-1, -1), border_width, colors.black)
        style.add('BOX', (0, 0), (-1, -1), border_width, colors.black)

        report.append_table(header, content, style)


class DistributionPlotsEntry(ReportEntry):

    def __init__(self, **params):
        super(DistributionPlotsEntry, self).__init__(**params)
        self._data = params.get('data', pd.DataFrame())
        self._column = params.get('column', '')
        self._by = params.get('by', '')

    def append_to_report(self, report):
        if self._data.empty:
            return

        df = self._data
        fig = plt.figure(figsize=(6, 6))

        hist_ax = fig.add_subplot(211)
        data_column = df[self._column]
        minimal, maximal = data_column.min(), data_column.max()
        hist_ax.hist(data_column, bins=10, range=(minimal, maximal))
        spaced = self._column.replace("_", " ")
        title = spaced.title() + " distribution"
        hist_ax.set_title(title)
        hist_ax.set_xlabel(spaced.title())
        hist_ax.set_ylabel("Count of Passengers")

        if self._by:
            boxplot_ax = fig.add_subplot(212)
            df.boxplot(ax=boxplot_ax, column=self._column, by=self._by)
            hist_ax.set_ylabel("Count of Passengers")

        fig.suptitle("")

        image_bytes = io.BytesIO()
        fig.tight_layout()
        fig.savefig(image_bytes)

        report.append_image(image_bytes)


class SimpleDataExplorer:
    """
    Calculates statistics on provided data set.
    """
    def __init__(self, **params):
        self._raw_data = pd.DataFrame()
        self._columns_slice = []
        self._report_file = ''
        self._boxplot_group = ''
        self._parse_args(**params)

    def _parse_args(self, **params):
        if 'data_file' in params and 'source' in params:
            raise ValueError("Parameters 'file_name' and 'source' cannot be"
                             "provided simultaneously")

        file_name = params.get('data_file', '')
        if file_name:
            reader_config = params.get('source_config', {})
            source = CsvDataSource(file_name, reader_config=reader_config)
            source.prepare()
        else:
            source = params['source']

        raw_data = source.data
        columns_slice = params.get('columns_slice', [])
        if columns_slice == 'all':
            columns_slice = list(source.data.columns)

        self._raw_data = raw_data
        self._columns_slice = columns_slice
        self._boxplot_group = params.get('boxplot_group', None)

        prefix = os.path.splitext(file_name)[0]
        default_report_name = prefix + "_report.pdf"
        self._report_file = params.get('report_file', default_report_name)

    @property
    def plotted_slice(self):
        columns = self._columns_slice
        if self._boxplot_group:
            columns.append(self._boxplot_group)
        return self._raw_data[columns]

    @property
    def dataset(self):
        return self._raw_data

    @property
    def report_file(self):
        return self._report_file

    @report_file.setter
    def report_file(self, value: str):
        self._report_file = value

    def generate_report(self):
        report = DataExplorationReport(self._report_file)
        stats = self._raw_data.describe().transpose()
        stats["median"] = self._raw_data.median()

        # add total records
        text = "Records count: " + str(stats["count"][0])
        report.add_element(PlainTextEntry(text=text))
        del stats["count"]

        # add stats table
        df_without_index = stats.reset_index()
        report.add_element(TextualTableEntry(table=df_without_index))

        # add distribution plots
        sliced_data = self.plotted_slice
        for column in self._columns_slice:
            plots = DistributionPlotsEntry(
                data=sliced_data, column=column, by=self._boxplot_group)
            report.add_element(plots)

        # save report into file system
        report.generate()
