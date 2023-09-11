#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 11-09-2023 18:24:14

import json
from typing import Dict, Union

import numpy as np

from ...utils import Role
from .... import constants as cs
from .utils import after_ditribution, distribute
from ...utils_mpi import MC, MPI_TAGS


def group_run(sts: MC, params: Dict, nv: int):
    mpi_comm, mpi_size = sts.mpi_comm, sts.mpi_size

    sts.logger.info("Distribution")
    sts.logger.debug("Rank 1: csvWriter")
    mpi_comm.send(obj=Role.csvWriter, dest=1, tag=MPI_TAGS.DISTRIBUTION)
    sts.logger.debug("Rank 2: adios_writer")
    mpi_comm.send(obj=Role.adios_writer, dest=2, tag=MPI_TAGS.DISTRIBUTION)

    thread_len = 3
    thread_num = round(np.floor((mpi_size - nv) / thread_len))
    sts.logger.info(f"Mpi_size: {mpi_size}, service threads: {nv}, thread length: {thread_len}, number of threads: {thread_num}")
    sts.logger.info(f"Total used workers: {thread_num * thread_len}")

    sts.logger.info("Sending info about roles")
    for i in range(thread_num):
        for j in range(thread_len):
            wrank = thread_len*i + j + nv
            if j % thread_len == 0:
                sts.logger.debug(f"Rank {wrank}, reader")
                mpi_comm.send(obj=Role.reader, dest=wrank, tag=MPI_TAGS.DISTRIBUTION)
            elif j % thread_len == 1:
                sts.logger.debug(f"Rank {wrank}, proceeder")
                mpi_comm.send(obj=Role.proceeder, dest=wrank, tag=MPI_TAGS.DISTRIBUTION)
            elif j % thread_len == 2:
                sts.logger.debug(f"Rank {wrank}, treater")
                mpi_comm.send(obj=Role.treater, dest=wrank, tag=MPI_TAGS.DISTRIBUTION)

    # readers = [nv + thread_len*i for i in range(thread_num)]
    # proceeders = [nv + thread_len*i + 1 for i in range(thread_num)]
    # treaters = [nv + thread_len*i + 2 for i in range(thread_num)]

    # [mpi_comm.send(obj=Role.reader,    dest=i, tag=MPI_TAGS.DISTRIBUTION) for i in readers]
    # [mpi_comm.send(obj=Role.proceeder, dest=i, tag=MPI_TAGS.DISTRIBUTION) for i in proceeders]
    # [mpi_comm.send(obj=Role.treater,   dest=i, tag=MPI_TAGS.DISTRIBUTION) for i in treaters]

    sts.logger.info(f"Killing remaining {mpi_size - (thread_len * thread_num + nv)} threads")
    for i in range(thread_len * thread_num + nv, mpi_size):
        sts.logger.debug(f"Rank {i}, killed")
        mpi_comm.send(obj=Role.killed, dest=i, tag=MPI_TAGS.DISTRIBUTION)

    sts.logger.debug("Sending list of threads to wait data from to csvWriter")
    mpi_comm.send(obj=[nv + 2 + thread_len*i for i in range(thread_num)], dest=1, tag=MPI_TAGS.TO_ACCEPT)
    sts.logger.debug("Sending list of threads to wait data from to adios_writer")
    mpi_comm.send(obj=[nv + 1 + thread_len*i for i in range(thread_num)], dest=2, tag=MPI_TAGS.TO_ACCEPT)

    sts.logger.debug("Sending processing folder to csvWriter")
    mpi_comm.send(obj=params[cs.fields.data_processing_folder], dest=1, tag=MPI_TAGS.SERV_DATA)
    sts.logger.debug("Sending processing folder to adios_writer")
    mpi_comm.send(obj=params[cs.fields.data_processing_folder], dest=2, tag=MPI_TAGS.SERV_DATA)

    sts.logger.info("Distributing storages")
    wd: Dict[str, Dict[str, Union[int, Dict[str, int]]]] = distribute(params[cs.fields.storages], thread_num)
    sts.logger.debug(json.dumps(wd, indent=4))

    sts.logger.info("Sending needed data for workers")
    for i in range(thread_num):
        # sts.logger.debug("Sending storage for ")
        mpi_comm.send(obj=wd[str(i)], dest=nv + thread_len * i, tag=MPI_TAGS.SERV_DATA)  # storages for readers
        mpi_comm.send(obj=(params[cs.fields.N_atoms], params[cs.fields.dimensions]), dest=nv + thread_len * i + 1, tag=MPI_TAGS.SERV_DATA)  # data for proceeders
        mpi_comm.send(obj=params, dest=nv + thread_len * i + 2, tag=MPI_TAGS.SERV_DATA)  # something for proceeders

    # [mpi_comm.send(obj=wd[str(i)], dest=i, tag=MPI_TAGS.SERV_DATA) for i in readers]
    # [mpi_comm.send(obj=(params[cs.fields.N_atoms], params[cs.fields.dimensions]), dest=i, tag=MPI_TAGS.SERV_DATA) for i in proceeders]
    # [mpi_comm.send(obj=params, dest=i, tag=MPI_TAGS.SERV_DATA) for i in treaters]

    sts.logger = sts.logger.getChild('after_distrib')
    return after_ditribution(sts, nv)
