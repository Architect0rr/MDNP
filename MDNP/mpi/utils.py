#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 16-09-2023 18:56:59

from enum import Enum
from pathlib import Path
from typing import Union, List, Tuple, Dict

import adios2
import numpy as np
from numpy import typing as npt
from mpi4py import MPI

from .. import constants as cs

MPIComm = Union[MPI.Intracomm, MPI.Intercomm]
GatherResponseType = List[Tuple[str, int]]


class Role(str, Enum):
    def __init__(self, s: str) -> None:
        super().__init__()

    reader = 'reader'
    proceeder = 'proceeder'
    treater = 'treater'
    killed = 'killed'
    one_thread = 'one_threaded'
    csvWriter = 'csvWriter'
    adios_writer = 'adios_writer'
    matr = 'matr'


class COMMAND(int, Enum):
    def __init__(self, i: int) -> None:
        super().__init__()

    EXIT = -1


class STATE(int, Enum):
    def __init__(self, i: int) -> None:
        super().__init__()

    EXITED = -1
