"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import warnings
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from typing import List
import os
import torch
from pymatgen.core import Structure


class MP20(InMemoryDataset):
    def __init__(self, root: str, cache: bool = True):
        self.root = root
        self.raw_csv = os.path.join(root, "raw/all.csv")
        self.cache_path = os.path.join(root, "raw/all.pt")

        if cache and os.path.exists(self.cache_path):
            self._data = torch.load(self.cache_path, weights_only=False)


    def __getitem__(self, idx: int) -> Structure:
        return Structure.from_str(self._data[idx]["cif"], fmt="cif")

    def __len__(self) -> int:
        return len(self._data)