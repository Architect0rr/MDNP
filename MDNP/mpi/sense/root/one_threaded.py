#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 10-09-2023 07:22:45

import json
from typing import Dict, Union

from ...utils import Role
from .... import constants as cs
from .utils import after_ditribution, distribute
from ...utils_mpi import MC, MPI_TAGS


def one_threaded(sts: MC, params: Dict, nv: int):
    mpi_comm, mpi_size = sts.mpi_comm, sts.mpi_size

    thread_num = mpi_size - nv
    print(f"Thread num: {thread_num}")
    wd: Dict[str, Dict[str, Union[int, Dict[str, int]]]] = distribute(params[cs.fields.storages], thread_num)
    print("Distribution")
    print(json.dumps(wd, indent=4))

    for i in range(thread_num):
        mpi_comm.send(obj=Role.one_thread, dest=i + nv, tag=MPI_TAGS.DISTRIBUTION)

    for i in range(thread_num):
        mkl = (wd[str(i)][cs.fields.number], wd[str(i)][cs.fields.storages])
        mpi_comm.send(obj=mkl, dest=nv + i, tag=MPI_TAGS.SERV_DATA_1)
        mpi_comm.send(obj=params, dest=nv + i, tag=MPI_TAGS.SERV_DATA_2)

    return after_ditribution(sts, 1)
