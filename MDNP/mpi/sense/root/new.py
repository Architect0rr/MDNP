#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 15-09-2023 23:45:27

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Any

import adios2  # type: ignore
import numpy as np

from ...utils import Role
from .... import constants as cs
from .utils import distribute, gw2c
from ...utils_mpi import MC, MPI_TAGS


def gen_matrix(cwd: Path, params: Dict, storages: List[Path], cut: int, logger: logging.Logger):
    output_csv_fp: Path = cwd / params[cs.fields.data_processing_folder] / cs.files.cluster_distribution_matrix
    logger.debug(f"Trying to open {output_csv_fp.as_posix()}")
    with output_csv_fp.open("w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        logger.debug("Starting loop")
        for storage in storages:
            with adios2.open(storage.as_posix(), 'r') as reader:  # type: ignore
                for step in reader:
                    stee: int = step.read(cs.lcf.mat_step)
                    dist = step.read(cs.lcf.mat_dist)
                    writer.writerow(np.hstack([stee, dist[:cut]]).astype(dtype=np.uint32).flatten())

    logger.debug("Success")


def after_new(sts: MC, nv: int, params: Dict[str, Any]):
    cwd, mpi_comm, mpi_size = sts.cwd, sts.mpi_comm, sts.mpi_size

    gw2c(sts, nv)

    sts.logger.info("Gathering info about new matrix storages")
    storages = []
    max_sizes = []
    for i in range(nv, mpi_size):
        storage: Path
        max_cluster_size: int
        storage, max_cluster_size = mpi_comm.recv(source=i, tag=MPI_TAGS.SERV_DATA_3)
        storages.append((i, storage))
        max_sizes.append(max_cluster_size)

    storages.sort(key=lambda x: x[1])
    _storages = [storage[1] for storage in storages]

    # with open(data_file, 'r') as fp:
    #     son: Dict[str, Any] = json.load(fp)

    params[cs.fields.matrix_storages] = [storage.as_posix() for storage in _storages]

    sts.logger.info("Writing storages to datafile")
    data_file = (cwd / cs.files.data)
    with open(data_file, 'w') as fp:
        json.dump(params, fp)

    sts.logger.info("Generating csv matrix")
    gen_matrix(cwd, params, _storages, max(max_sizes), sts.logger.getChild('mtrix_gen'))

    sts.logger.info("Exiting...")
    return 0


def new(sts: MC, params: Dict, nv: int):
    mpi_comm, mpi_size = sts.mpi_comm, sts.mpi_size

    thread_num = mpi_size - nv
    sts.logger.info(f"Workers count: {thread_num}")

    sts.logger.info("Sending info about roles")
    for i in range(thread_num):
        mpi_comm.send(obj=Role.matr, dest=i + nv, tag=MPI_TAGS.DISTRIBUTION)

    sts.logger.info("Distributing storages")
    wd: Dict[str, Dict[str, Union[int, Dict[str, int]]]] = distribute(params[cs.fields.storages], thread_num)
    sts.logger.debug(json.dumps(wd, indent=4))

    sts.logger.info("Sending needed data for workers")
    for i in range(thread_num):
        mkl = (wd[str(i)][cs.fields.number], wd[str(i)][cs.fields.storages])
        mpi_comm.send(obj=mkl, dest=nv + i, tag=MPI_TAGS.SERV_DATA_1)
        mpi_comm.send(obj=params, dest=nv + i, tag=MPI_TAGS.SERV_DATA_2)

    sts.logger = sts.logger.getChild('after_new')
    return after_new(sts, nv, params)
