"""
This script is only for test.
Not the part of the project.
"""

from importlib import import_module
import argparse
import threading
import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--clear', dest='if_restart', action='store_const',
                    const=True, default=False,
                    help='if true, then restart training')
args = parser.parse_args()
print args.if_restart