# This module separates out the logging functionality from the geralbaldi.sim module
# By doing so, the logging functionality can be tested individually without having to consider the rest of the geralbaldi.sim.simulate() function
import sys
import os
import logging

# Add the directory containing the gerabaldi module to the Python module search path
gerabaldi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, gerabaldi_path)
from gerabaldi.helpers import logger

def simulate_test(logging_level = None, file_handler = None, stream_handler = None):
    def _configure_logger():
        """Configure the logger based on the user's preference."""
        default_stream_handler = None
        default_file_handler = None

        # If the user has provided a global logging level, set accordingly
        if logging_level:
            logger.setLevel(logging_level)

        # Identify the default handlers based on their types
        # NOTE: The assumption here is that the logger has one stream and one file handler maximum
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                default_stream_handler = handler
            elif isinstance(handler, logging.FileHandler):
                default_file_handler = handler
        
        # If the user provided a custom stream handler, remove the default one, add the custom one
        if stream_handler:
            # If there's a default stream handler, remove it first
            if default_stream_handler:
                logger.removeHandler(default_stream_handler)
            logger.addHandler(stream_handler)
        
        # If the user provided a custom file handler, remove the default one, add the custom one
        if file_handler:
            if default_file_handler:
                logger.removeHandler(default_file_handler)
            logger.addHandler(file_handler)
        

    _configure_logger()
    logger.info("Simulating...")