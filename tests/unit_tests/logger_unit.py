# This module separates out the logging functionality from the geralbaldi.sim module
# By doing so, the logging functionality can be tested individually without having to consider the rest of the geralbaldi.sim.simulate() function
import sys
import os
# Add the directory containing the gerabaldi module to the Python module search path
gerabaldi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, gerabaldi_path)
from gerabaldi.helpers import logger


def simulate_test():
    logger.info("Simulating...")