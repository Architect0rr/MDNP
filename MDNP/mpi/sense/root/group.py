#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 10-09-2023 07:21:21

import json
from typing import Dict, Union

import numpy as np

from ...utils import Role
from .... import constants as cs
from .utils import after_ditribution, distribute
from ...utils_mpi import MC, MPI_TAGS


def group_run(sts: MC, params: Dict, nv: int):
    mpi_comm, mpi_size = sts.mpi_comm, sts.mpi_size

    mpi_comm.send(obj=Role.csvWriter, dest=1, tag=MPI_TAGS.DISTRIBUTION)
    mpi_comm.send(obj=Role.adios_writer, dest=2, tag=MPI_TAGS.DISTRIBUTION)

    thread_len = 3
    thread_num = round(np.floor((mpi_size - nv) / thread_len))
    print(f"Thread num: {thread_num}")
    wd: Dict[str, Dict[str, Union[int, Dict[str, int]]]] = distribute(params[cs.fields.storages], thread_num)
    print("Distribution")
    print(json.dumps(wd, indent=4))

    for i in range(thread_num):
        for j in range(thread_len):
            mpi_comm.send(obj=Role(j), dest=thread_len*i + j + nv, tag=MPI_TAGS.DISTRIBUTION)

    for i in range(thread_len * thread_num + nv, mpi_size):
        mpi_comm.send(obj=Role.killed, dest=i, tag=MPI_TAGS.DISTRIBUTION)
    mpi_comm.send(obj=[nv + 1 + thread_len*i for i in range(thread_num)], dest=2, tag=MPI_TAGS.TO_ACCEPT)
    mpi_comm.send(obj=[nv + 2 + thread_len*i for i in range(thread_num)], dest=1, tag=MPI_TAGS.TO_ACCEPT)

    mpi_comm.send(obj=params[cs.fields.data_processing_folder], dest=2, tag=MPI_TAGS.SERV_DATA)
    mpi_comm.send(obj=params[cs.fields.data_processing_folder], dest=1, tag=MPI_TAGS.SERV_DATA)

    for i in range(thread_num):
        mpi_comm.send(obj=wd[str(i)], dest=nv + thread_len * i, tag=MPI_TAGS.SERV_DATA)
        mpi_comm.send(obj=params[cs.cf.dump_folder], dest=nv + thread_len * i, tag=MPI_TAGS.SERV_DATA_1)
        mpi_comm.send(obj=(params[cs.fields.N_atoms], params[cs.fields.dimensions]), dest=nv + thread_len * i + 1, tag=MPI_TAGS.SERV_DATA)
        mpi_comm.send(obj=params, dest=nv + thread_len * i + 2, tag=MPI_TAGS.SERV_DATA)

    return after_ditribution(sts, 3)