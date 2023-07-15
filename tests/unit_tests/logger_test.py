import sys
import os
import logging
from unittest import TestCase
from io import StringIO

# Add the directory containing the gerabaldi module to the Python module search path
gerabaldi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, gerabaldi_path)
from gerabaldi.helpers import logger

class TestLogger(TestCase):
    def test_logger_name(self):
        """Test that the logger has the correct name"""
        with self.assertLogs() as captured_logs:
            logger.info("Testing logger's name.")
        self.assertEqual(captured_logs.records[0].name, "gerabaldi.helpers")

    def test_logger_level(self):
        """Test that the correct logging level has been used"""
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
        """Test that the correct log message has been delivered"""
        with self.assertLogs() as captured_logs:
            logger.warning("Test test test")
        self.assertEqual(captured_logs.records[0].getMessage(), "Test test test")
    
    def test_logger_format(self):
        """Test that the handlers have the correct format"""
        for handler in logger.handlers:
            self.assertEqual(handler.formatter._fmt, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            




#t = TestLogger()
#t.test_logger_format()