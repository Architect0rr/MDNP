#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 10-09-2023 00:59:50

from enum import Enum
from typing import Union, List, Tuple

from mpi4py import MPI


MPIComm = Union[MPI.Intracomm, MPI.Intercomm]
GatherResponseType = List[Tuple[str, int]]


class Role(str, Enum):
    reader = 'reader'
    proceeder = 'proceeder'
    treater = 'treater'
    killed = 'killed'
    one_thread = 'one_threaded'
    csvWriter = 'csvWriter'
    adios_writer = 'adios_writer'
    matr = 'matr'
