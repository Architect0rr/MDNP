#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 11-09-2023 18:14:22

import json
from typing import Dict, Union

from ...utils import Role
from .... import constants as cs
from .utils import after_ditribution, distribute
from ...utils_mpi import MC, MPI_TAGS


def one_threaded(sts: MC, params: Dict, nv: int):
    mpi_comm, mpi_size = sts.mpi_comm, sts.mpi_size

    thread_num = mpi_size - nv
    sts.logger.info(f"Workers count: {thread_num}")

    sts.logger.info("Sending info about roles")
    for i in range(thread_num):
        mpi_comm.send(obj=Role.one_thread, dest=i + nv, tag=MPI_TAGS.DISTRIBUTION)

    sts.logger.info("Distributing storages")
    wd: Dict[str, Dict[str, Union[int, Dict[str, int]]]] = distribute(params[cs.fields.storages], thread_num)
    sts.logger.debug(json.dumps(wd, indent=4))

    sts.logger.info("Sending needed data for workers")
    for i in range(thread_num):
        mkl = (wd[str(i)][cs.fields.number], wd[str(i)][cs.fields.storages])
        mpi_comm.send(obj=mkl, dest=nv + i, tag=MPI_TAGS.SERV_DATA_1)
        mpi_comm.send(obj=params, dest=nv + i, tag=MPI_TAGS.SERV_DATA_2)

    sts.logger = sts.logger.getChild('after_distrib')
    return after_ditribution(sts, nv)
