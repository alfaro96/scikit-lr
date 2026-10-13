"""Download the datasets of every paper of the reproduction to ``DATA_HOME``.

The jobs that run the experiments restore them from the cache, instead of each
downloading them from OpenML.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path("reproducibility", "workflow").resolve()))

from experiments import PAPERS, load_dataset

for paper in PAPERS.values():
    tag = paper["datasets"]["tag"]
    for name in paper["datasets"]["names"]:
        print(f"Downloading {name} ({tag})")
        load_dataset(tag, name, os.environ["DATA_HOME"])
