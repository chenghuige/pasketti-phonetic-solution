"""``gezi.common`` kitchen-sink — re-exports the symbols every project
file expects to receive from ``from gezi.common import *``.

Keeping this module in a fixed shape avoids touching the existing
``dataset.py`` / ``models/*.py`` / ``submit.py`` source files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# --- stdlib ---------------------------------------------------------------
import sys
import os
import io
import re
import gc
import json
import math
import time
import copy
import glob
import gzip
import shutil
import pickle
import random
import warnings
import subprocess
import collections
import itertools
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from multiprocessing import Pool, Manager, cpu_count

# --- numerical / data ----------------------------------------------------
import numpy as np
import pandas as pd  # noqa: F401 — used by submit.py / ensemble.py
import scipy           # noqa: F401
import sklearn         # noqa: F401
from sklearn.preprocessing import normalize  # noqa: F401

# --- pytorch (heavily used by models / dataset) --------------------------
try:
  import torch                     # noqa: F401
  import torch.nn as nn            # noqa: F401
  import torch.nn.functional as F  # noqa: F401
except Exception:                  # pragma: no cover
  torch = None                     # type: ignore[assignment]
  nn = None                        # type: ignore[assignment]
  F = None                         # type: ignore[assignment]

# --- progress & logging --------------------------------------------------
from tqdm.auto import tqdm  # noqa: F401

# --- absl flags (the project uses ``FLAGS`` directly) --------------------
from absl import app, flags  # noqa: F401
FLAGS = flags.FLAGS

# --- our own shim packages re-exported under the historical aliases ------
import gezi  # this same package
import gezi as gz  # noqa: F401  — gz.* alias
from gezi import (  # noqa: F401
    logger,
    ic,
    ic_once,
    ic_nth,
    ico,
    icn,
    gic,
    dic,
    icl,
    rtqdm,
    Globals,
    Globals as GLBS,
    Timer,
    tree,
)
import melt           # noqa: F401
import melt as mt     # noqa: F401
try:
  import lele
  import lele as le   # noqa: F401
except Exception:     # pragma: no cover
  lele = None         # type: ignore[assignment]
  le = None           # type: ignore[assignment]
import husky          # noqa: F401  — placeholder

# Dotted attribute for ``gezi.common.PERCENTILES`` style lookups in the
# upstream code (rarely used here, but kept for safety).
PERCENTILES = [.25, .5, .75, .9, .95, .99]
SPECIAL_CHAR = 'ʶ'
SPECIAL_EN = '。'
FAIL = '❌'
PASS = '✅'
WARNING = '⚠️'
STAR = '⭐'
FIRE = '🔥'
TABLE = '📋'
CHART = '📊'
SEARCH = '🔍'

# Some modules expect a `set` callable; restore the builtin to avoid
# accidental shadowing by ``gezi.set``.
from builtins import set  # noqa: F401, A004
