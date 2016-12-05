"""
Supplement classes to utilize builtin logging module.
"""

from ..config import LoggingParameters


class FileLogHandler:
    """
    Handler for logger that writes debug and error messages into separate text
    files.
    """

    splitter = "::"
    format = "%(levelname)s{0}%(asctime)s{0}%(message)s\n".format(splitter)
    files = {
        'ERROR': LoggingParameters.ERROR_OUTPUT_FILE,
        'INFO': LoggingParameters.REGULAR_OUTPUT_FILE,
    }

    def write(self, msg):
        log_file = FileLogHandler._get_file(msg)
        if not log_file:
            return
        with open(log_file, "a") as f:
            f.write(msg)

    @staticmethod
    def _get_file(msg):
        level, *_ = msg.split(FileLogHandler.splitter)
        return FileLogHandler.files.get(level, "")

