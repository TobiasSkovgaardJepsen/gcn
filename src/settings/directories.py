""" Provides easy access to data and models directories. """
from dotenv import load_dotenv, find_dotenv
from os.path import dirname, abspath, join as path_join

# Initialize environment variables from .env file in (parent) directory
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = dirname(abspath(dotenv_path))

MODELS_DIR = path_join(PROJECT_DIR, 'models')

# Data Directories
DATA_DIR = path_join(PROJECT_DIR, 'data')
RAW_DATA_DIR = path_join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = path_join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = path_join(DATA_DIR, 'processed')
