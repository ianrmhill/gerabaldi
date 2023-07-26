import sys
import os
import logging
from io import StringIO
from unittest import TestCase
# Add the directory containing the gerabaldi module to the Python module search path
gerabaldi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, gerabaldi_path)
from gerabaldi.helpers import logger
from tests.unit_tests.logger_unit import simulate_test


class TestLogger(TestCase):
    def test_logger_name(self):
        """Test that the logger has the correct name."""
        with self.assertLogs() as captured_logs:
            logger.info("Testing logger's name.")
        self.assertEqual(captured_logs.records[0].name, "gerabaldi.helpers")

    def test_logger_level(self):
        """Test that the correct logging level has been used."""
        logger.debug("Debug")
        # Check that nothing has been logged at the debug level
        has_debug_logs = False
        for handler in logger.handlers:
            if hasattr(handler, 'buffer') and any(record.levelno == logging.DEBUG for record in handler.buffer):
                has_debug_logs = True
                break
        self.assertFalse(has_debug_logs, "Debug message was logged")

        with self.assertLogs() as captured_logs:
            logger.info("Info")
        self.assertEqual(captured_logs.records[0].levelname, "INFO")

        with self.assertLogs() as captured_logs:
            logger.warning("Warning")
        self.assertEqual(captured_logs.records[0].levelname, "WARNING")

        with self.assertLogs() as captured_logs:
            logger.error("Error")
        self.assertEqual(captured_logs.records[0].levelname, "ERROR")

        with self.assertLogs() as captured_logs:
            logger.critical("Critical")
        self.assertEqual(captured_logs.records[0].levelname, "CRITICAL")

    def test_log_message(self):
        """Test that the correct log message has been delivered."""
        with self.assertLogs() as captured_logs:
            logger.warning("Test test test")
        self.assertEqual(captured_logs.records[0].getMessage(), "Test test test")

    def test_logger_format(self):
        """Test that the handlers have the correct format"""
        for handler in logger.handlers:
            self.assertEqual(handler.formatter._fmt, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def test_simulate_logging(self):
        """Test the logging functionality of the geralbaldi.sim.simulate() function."""
        def get_stream_handler():
            log_stream = StringIO()
            my_stream_handler = logging.StreamHandler(stream=log_stream)

            return log_stream, my_stream_handler

        # There should be no logging messages, as the stream handler's level is set to WARNING
        log_stream, my_stream_handler = get_stream_handler()
        my_stream_handler.setLevel(logging.WARNING)
        simulate_test(stream_handler=my_stream_handler)
        my_stream_handler.flush()
        log_output = log_stream.getvalue()
        self.assertEqual(len(log_output), 0)

        # Now with the level set to INFO, the message should be "Simulating..."
        log_stream, my_stream_handler = get_stream_handler()
        my_stream_handler.setLevel(logging.INFO)
        simulate_test(stream_handler=my_stream_handler)
        my_stream_handler.flush()
        log_output = log_stream.getvalue()
        self.assertEqual(log_output.rstrip(), "Simulating...")

        # Now set the global logger level to WARNING, there should be no messages
        log_stream, my_stream_handler = get_stream_handler()
        my_stream_handler.setLevel(logging.INFO)
        simulate_test(logging_level=logging.WARNING, stream_handler=my_stream_handler)
        my_stream_handler.flush()
        log_output = log_stream.getvalue()
        self.assertEqual(len(log_output), 0)

        # Now set the global logger level to INFO, the message should be "Simulating..."
        log_stream, my_stream_handler = get_stream_handler()
        my_stream_handler.setLevel(logging.INFO)
        simulate_test(logging_level=logging.INFO, stream_handler=my_stream_handler)
        my_stream_handler.flush()
        log_output = log_stream.getvalue()
        self.assertEqual(log_output.rstrip(), "Simulating...")

        # Test the file handler
        my_file_handler = logging.FileHandler("test.log")
        my_file_handler.setLevel(logging.INFO)
        simulate_test(file_handler=my_file_handler)
        with open("test.log", "r") as file:
            log_content = file.read().rstrip()
        os.remove("test.log")
        self.assertEqual(log_content, "Simulating...")

        # Set the file handler level to WARNING, there should be nothing in the log
        my_file_handler = logging.FileHandler("test.log")
        my_file_handler.setLevel(logging.WARNING)
        simulate_test(file_handler=my_file_handler)
        with open("test.log", "r") as file:
            log_content = file.read().rstrip()
        os.remove("test.log")
        self.assertEqual(log_content, "")