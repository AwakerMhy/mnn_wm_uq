"""Logger."""

import logging


class Logger:
    def __init__(self, log_file: str = "attractor.log", loger_name: str = "mnn attractor"):
        """Initialize the logger.

        Args:
        ----
            log_file (string, optional): The file to log to.

        """
        super().__init__()

        self.logger = logging.getLogger(loger_name)
        self.logger.setLevel(logging.INFO)

        # Create a file handler
        self.file_handler = logging.FileHandler(log_file)

        # Create a stream handler (prints to console)
        self.stream_handler = logging.StreamHandler()

        # Create a formatter
        self.formatter = logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")

        # Set the formatter for both handlers
        self.file_handler.setFormatter(self.formatter)
        self.stream_handler.setFormatter(self.formatter)

        # Add the handlers to the logger
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stream_handler)
        self.logger.propagate = False

    def log(self, message: str):
        """Log a message.

        Args:
        ----
            message (string): The message to log.

        """
        self.logger.info(message)
