"""WikiNews NLP analysis package.

Importing this package caps the thread budget of the numerical libraries. It
has to happen here, before anything pulls in torch or tokenizers, because those
libraries read the variables once at import time.
"""

from __future__ import annotations

import os

_THREAD_LIMIT = os.environ.get("NLP_THREAD_LIMIT", "2")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _THREAD_LIMIT)

# Stops the HuggingFace tokenizers forking worker processes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__all__ = ["__version__"]
__version__ = "0.1.0"
