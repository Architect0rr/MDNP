#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 11-09-2023 20:30:50

import csv
from typing import List

import adios2  # type: ignore
import numpy as np
from numpy import typing as npt

from ...utils_mpi import MC, MPI_TAGS
from .... import constants as cs


def adios_writer(sts: MC):
    cwd, mpi_comm = sts.cwd, sts.mpi_comm

    sts.logger.info("Receiving acceptable list")
    threads: List[int] = mpi_comm.recv(source=0, tag=MPI_TAGS.TO_ACCEPT)
    sts.logger.info("List received")
    sts.logger.info("Receiving folder")
    folder: str = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA)
    sts.logger.info("Folder received")

    sts.logger.info("Creating storage")
    with adios2.open((cwd / folder / cs.files.mat_storage).as_posix(), 'w') as adout:  # type: ignore
        sts.logger.info("Starting main loop")
        while any([mpi_comm.iprobe(source=i, tag=MPI_TAGS.WRITE) for i in threads]) and not mpi_comm.iprobe(source=0, tag=MPI_TAGS.COMMAND):
            for thread in threads:
                if mpi_comm.iprobe(source=thread, tag=MPI_TAGS.WRITE):
                    step: int
                    arr: npt.NDArray[np.float32]
                    step, arr = mpi_comm.recv(source=thread, tag=MPI_TAGS.WRITE)
                    adout.write(cs.lcf.mat_step, np.array(step))  # type: ignore
                    adout.write(cs.lcf.mat_dist, arr, arr.shape, np.full(len(arr.shape), 0), arr.shape, end_step=True)  # type: ignore
                    mpi_comm.send(obj=step, dest=0, tag=MPI_TAGS.STATE)

    sts.logger.info("Exiting...")
    return 0


def csvWriter(sts: MC):
    cwd, mpi_comm = sts.cwd, sts.mpi_comm

    sts.logger.info("Receiving acceptable list")
    threads: List[int] = mpi_comm.recv(source=0, tag=MPI_TAGS.TO_ACCEPT)
    sts.logger.info("List received")
    sts.logger.info("Receiving folder")
    folder: str = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA)
    sts.logger.info("Folder received")

    sts.logger.info("Creating output file")
    ctr: int = 0
    with open((cwd / folder / cs.files.comp_data), "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        sts.logger.info("Starting main loop")
        while any([mpi_comm.iprobe(source=i, tag=MPI_TAGS.WRITE) for i in threads]) and not mpi_comm.iprobe(source=0, tag=MPI_TAGS.COMMAND):
            for thread in threads:
                if mpi_comm.iprobe(source=thread, tag=MPI_TAGS.WRITE):
                    data: npt.NDArray[np.float32] = mpi_comm.recv(source=thread, tag=MPI_TAGS.WRITE)
                    writer.writerow(data)
                    ctr += 1
                    mpi_comm.send(obj=ctr, dest=0, tag=MPI_TAGS.STATE)
                    csv_file.flush()

    sts.logger.info("Exiting...")
    return 0
