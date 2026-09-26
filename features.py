"""
features.py
Importable module for pairwise similarity feature computation.
(Same content as src/03_features.py — kept as top-level for clean imports in run_pipeline.py)
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from features_core import *
