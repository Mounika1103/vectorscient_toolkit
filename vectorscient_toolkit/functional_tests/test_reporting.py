import pandas as pd

from ..pdf.entry import ListOfItemsEntry
from ..pdf.report import PDFReport


class TestReportGeneration:

    def test_list_of_items_report_entry(self):
        report = PDFReport()
        ser = pd.Series({"First": 1, "Second": 2, "Third": 3})

        entry = ListOfItemsEntry(ser, header="Items with Values", width=400)
        report.add_entry_to_report(entry)
        report.save("report.pdf")
