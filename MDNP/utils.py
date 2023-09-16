#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 16-09-2023 19:00:16

from pathlib import Path
from typing import Union, Iterable, Tuple

import adios2
import numpy as np
from numpy import typing as npt

from . import constants as cs

fp = Union[float, np.floating]


def is_iter(arr: Union[Iterable[float], float]) -> bool:
    try:
        iter(arr)  # type: ignore
        return True
    except Exception:
        return False


def bearbeit(storage: Path) -> Tuple[int, npt.NDArray[np.float32]]:
    adin = adios2.open(storage.as_posix(), 'r')  # type: ignore

    N = int(adin.read(cs.lcf.natoms))
    Lx = float(adin.read(cs.lcf.boxxhi))
    Ly = float(adin.read(cs.lcf.boxyhi))
    Lz = float(adin.read(cs.lcf.boxzhi))

    adin.close()

    bdims = np.array([Lx, Ly, Lz])
    return (N, bdims)
