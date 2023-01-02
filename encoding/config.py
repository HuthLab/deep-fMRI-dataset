import os
from os.path import join, dirname



REPO_DIR = join(dirname(dirname(os.path.abspath(__file__))))
EM_DATA_DIR = join(REPO_DIR, 'em_data')
DATA_DIR = join(REPO_DIR, 'encoding', 'data')

