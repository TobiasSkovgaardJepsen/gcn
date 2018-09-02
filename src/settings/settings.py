""" Contains global project settings. """
from os import getenv
import logging

LOG_LEVEL = getenv('LOG_LEVEL', default='WARNING')
logging.basicConfig(
        level=getattr(logging, LOG_LEVEL)
)
