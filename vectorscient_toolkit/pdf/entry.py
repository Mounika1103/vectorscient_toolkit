from datetime import datetime
import io

from reportlab.platypus import TableStyle, Image, Table, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import PageBreak, PageTemplate, Frame
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..utils import res


class ReportEntry:
    """
    Base class for PDF report entries.

    Each subclass should implement a sequence of steps that create an piece of
    PDF document, i.e. image, table, paragraph of text, etc.
    """

    def get_content(self, doc):
        raise NotImplementedError()


class PlainTextEntry(ReportEntry):

    def __init__(self, text, space_before=None, space_after=None, style=None):
        self.text = text
        self.space_before = space_before
        self.space_after = space_after
        self.style = style

    def get_content(self, doc):
        if not self.text:
            return None
        styles = getSampleStyleSheet()
        style = styles["Normal" if not self.style else self.style]
        items = []
        if self.space_before is not None:
            items.append(Spacer(width=0, height=self.space_before))
        items.append(Paragraph(self.text, style))
        if self.space_after is not None:
            items.append(Spacer(width=0, height=self.space_after))
        return items


class PageHeader(ReportEntry):

    def __init__(self, header, bottom_padding=None):
        self.header = header
        self.bottom_padding = bottom_padding

    def get_content(self, doc):
        if not self.header:
            return None
        style = getSampleStyleSheet()["Normal"]
        style.fontSize = 12
        header = Paragraph(self.header, style)
        padding = self.bottom_padding if self.bottom_padding else 20
        space_after = Spacer(0, padding)
        return [header, space_after]


class TextualTableEntry(ReportEntry):

    def __init__(self, table_data, title='', subtitle='', **params):
        self.table = table_data
        self.title = title
        self.subtitle = subtitle
        self.full_width = params.get("full_width", 400)
        self.column_widths = params.get("column_widths", None)
        self.ignore_index = params.get("ignore_index", True)
        self.transpose = params.get("transpose", False)

    def get_content(self, doc):
        if self.table.empty:
            return None

        def to_fixed(row):
            updated = []
            for item in row:
                if isinstance(item, float):
                    item = str(round(item, ndigits=4))
                updated.append(item)
            new = pd.Series(updated)
            new.name = row.name
            return new

        df = self.table

        if not self.ignore_index:
            # create index column
            df.reset_index(inplace=True)

        if self.transpose:
            df = self.table.transpose()

        rounded = df.apply(to_fixed, axis=1)
        rounded.columns = df.columns
        header = rounded.columns.tolist()
        content = rounded.values.tolist()
        whole_table = [header] + content

        if self.column_widths is None:
            n = len(df.columns)
            self.column_widths = [self.full_width/n] * n

        table = Table(whole_table, colWidths=self.column_widths)

        style = TableStyle()
        border_width = 0.25
        style.add('INNERGRID', (0, 0), (-1, -1), border_width, colors.black)
        style.add('BOX', (0, 0), (-1, -1), border_width, colors.black)
        style.add('FONTSIZE', (0, 0), (-1, -1), 9)
        table.setStyle(style)

        return table


class ImageEntry(ReportEntry):

    def __init__(self, data, width=None, height=None):
        self.data = data
        self.width = width
        self.height = height

    def get_content(self, doc):
        img = ImageReader(self.data)
        sz = img.getSize()
        adj_w, adj_h = self._adjusted_image_size(
            actual=sz, adjusted=(self.width, self.height))
        img = Image(self.data, width=adj_w, height=adj_h)
        return img

    def _adjusted_image_size(self, actual, adjusted):
        current_w, current_h = actual
        w, h = adjusted

        if w is not None:
            ratio = w/float(current_w)
            new_w = w
            new_h = current_h * ratio

        elif h is not None:
            ratio = h/float(current_h)
            new_w = current_w * ratio
            new_h = h

        else:
            return actual

        return new_w, new_h


class DistributionPlotsEntry(ReportEntry):

    def __init__(self, data, column, by):
        self.data = data
        self.column = column
        self.by = by

    def get_content(self, doc):
        if self.data.empty:
            return

        df = self.data
        fig = plt.figure(figsize=(6, 6))

        hist_ax = fig.add_subplot(211)
        data_column = df[self.column]
        minimal, maximal = data_column.min(), data_column.max()
        hist_ax.hist(data_column, bins=10, range=(minimal, maximal))
        spaced = self.column.replace("_", " ")
        title = spaced.title() + " distribution"
        hist_ax.set_title(title)
        hist_ax.set_xlabel(spaced.title())
        hist_ax.set_ylabel("Count of Passengers")

        if self.by:
            boxplot_ax = fig.add_subplot(212)
            df.boxplot(ax=boxplot_ax, column=self.column, by=self.by)
            hist_ax.set_ylabel("Count of Passengers")

        fig.suptitle("")
        image_bytes = io.BytesIO()
        fig.tight_layout()
        fig.savefig(image_bytes)
        img = Image(image_bytes, width=300, height=300)
        return img


class BreakEntry(ReportEntry):

    def __init__(self, mode="pagebreak"):
        self.mode = mode

    def get_content(self, report):
        if self.mode != "pagebreak":
            return None
        return PageBreak()


class ListOfItemsEntry(ReportEntry):

    def __init__(self, items, header='', **params):
        if isinstance(items, pd.Series):
            self.items = [[k, v] for k, v in items.iteritems()]
        else:
            shape = list(np.shape(items))
            if len(shape) == 1:
                # list provided
                self.items = []
            else:
                # matrix provided
                self.items = items

        self.header = header
        self.width = params.get("width")
        even_color, odd_color = params.get(
            "even_odd_colors", (colors.lightskyblue, colors.aliceblue))
        self.even_color = even_color
        self.odd_color = odd_color
        self.alternate_colors = params.get("alternate_colors", True)
        self.border_color = params.get("border_color", colors.transparent)

    def get_content(self, doc):
        if len(self.items) == 0:
            return None

        style_commands = []

        if self.alternate_colors:
            for i, _ in enumerate(self.items):
                color = self.even_color if i % 2 == 0 else self.odd_color

                # Warning: it seems that some of reportlab docs have an error.
                # The styling command uses format (col, row), not vice versa.
                style_commands.append(
                    ("BACKGROUND", (0, i), (1, i), color))
                style_commands.append(
                    ("LINEBELOW", (0, i), (1, i), 1, self.border_color))

        style_commands.append(("ALIGN", (0, 0), (0, -1), "LEFT"))
        style_commands.append(("ALIGN", (1, 0), (1, -1), "RIGHT"))

        if self.width is None:
            self.width = doc.width

        list_of_items = Table(
            self.items,
            hAlign="LEFT",
            colWidths=[self.width / 2, self.width / 2],
            style=style_commands)

        if self.header:
            style = getSampleStyleSheet()["Normal"]
            style.fontSize = 12
            header = Paragraph(self.header, style)
            space_before = Spacer(0, 10)
            space_after = Spacer(0, 20)
            return [header, space_before, list_of_items, space_after]

        else:
            return list_of_items


class VectorScientHeadlineManager:
    def __init__(self, **params):
        self.page_count = 1
        self.disclaimer = params.get("disclaimer")

    def headline(self, canvas, doc):

        def draw_header():
            canvas.saveState()

            style = ParagraphStyle(name="ReportHeader",
                                   fontName="Helvetica",
                                   fontSize=16,
                                   textColor=colors.gray)
            caption = Paragraph("Prediction Classification Results", style)
            _, h = caption.wrap(doc.width, doc.height)
            full_width, full_height = doc.pagesize
            w = canvas.stringWidth(caption.text, style.fontName, style.fontSize)

            caption_x = (full_width - w)/2.0
            caption_y = full_height - 2*h
            caption.drawOn(canvas, caption_x, caption_y)

            canvas.setStrokeColor(colors.cornflowerblue)
            canvas.setLineWidth(1.5)
            line_x1 = 10
            line_x2 = full_width - line_x1
            line_y1 = line_y2 = caption_y - 30
            canvas.line(line_x1, line_y1, line_x2, line_y2)

            logo = Image(res("vs_logo.png"), width=108, height=28)
            logo.drawOn(canvas, doc.leftMargin/2, caption_y - h)

            website_name = "<u>www.VectorScient.com</u>"
            style = ParagraphStyle(name="ReportWebsite",
                                   fontName="Helvetica",
                                   fontSize=10,
                                   textColor=colors.cornflowerblue)
            website = Paragraph(website_name, style)
            website.wrap(doc.width, doc.height)
            website.drawOn(canvas, doc.leftMargin/2, caption_y - 2*h)

            canvas.restoreState()

        def draw_footer():
            canvas.saveState()

            style = ParagraphStyle(name="ReportFooter",
                                   fontName="Helvetica",
                                   fontSize=10,
                                   textColor=colors.gray)

            disclaimer = Paragraph(self.disclaimer, style)
            _, h = disclaimer.wrap(doc.width, doc.height)
            disclaimer.drawOn(canvas, doc.leftMargin, h - 10)

            publish_date = datetime.today().strftime("%m %h %Y / %H:%M")
            stamp_text = "{} / {}".format(publish_date, self.page_count)
            stamp = Paragraph(stamp_text, style)
            stamp_x, stamp_y = stamp.wrap(doc.width, doc.height)
            stamp.drawOn(canvas, stamp_x, stamp_y)

            full_width, full_height = doc.pagesize
            canvas.setStrokeColor(colors.cornflowerblue)
            canvas.setLineWidth(1.5)
            line_x1 = 10
            line_x2 = full_width - line_x1
            line_y1 = line_y2 = doc.bottomMargin
            canvas.line(line_x1, line_y1, line_x2, line_y2)

            canvas.restoreState()

        draw_header()
        draw_footer()
        self.page_count += 1

    def insert_into_report(self, report_template):
        doc = report_template
        frame = Frame(doc.leftMargin,
                      doc.bottomMargin,
                      doc.width,
                      doc.height,
                      id='normal')
        template = PageTemplate(id='vs_footer',
                                frames=frame,
                                onPage=self.headline)
        doc.addPageTemplates([template])
