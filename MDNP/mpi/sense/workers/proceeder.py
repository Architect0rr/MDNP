#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 11-09-2023 00:56:41


import os
import warnings
from typing import Literal


os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.simplefilter("error")


import freud
import numpy as np
from numpy import typing as npt

from ...utils import STATE
from ....core import distribution
from ...utils_mpi import MC, MPI_TAGS


def proceed(sts: MC) -> Literal[0]:
    mpi_comm, mpi_rank = sts.mpi_comm, sts.mpi_rank

    sts.logger.info("Reveiving data from root")
    N: int
    bdims: npt.NDArray[np.float32]
    N, bdims = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA)
    sts.logger.info("Data received")

    box = freud.box.Box.from_box(np.array(bdims))

    reader_rank = mpi_rank - 1
    trt_rank = mpi_rank + 1

    sts.logger.info("Starting main loop")
    while not mpi_comm.iprobe(source=reader_rank, tag=MPI_TAGS.SERVICE) or mpi_comm.iprobe(source=reader_rank, tag=MPI_TAGS.DATA):
        step: int
        sender: int
        data: npt.NDArray[np.float32]
        step, sender, data = mpi_comm.recv(source=reader_rank, tag=MPI_TAGS.DATA)
        if sender != reader_rank:
            pass

        dist = distribution.get_dist(data, N, box)

        tpl = (step, dist)

        mpi_comm.send(obj=tpl, dest=2, tag=MPI_TAGS.WRITE)
        mpi_comm.send(obj=tpl, dest=trt_rank, tag=MPI_TAGS.DATA)

        mpi_comm.send(obj=step, dest=reader_rank, tag=MPI_TAGS.SERVICE)

        mpi_comm.send(obj=step, dest=0, tag=MPI_TAGS.STATE)

        # if mpi_comm.iprobe(source=reader_rank, tag=MPI_TAGS.SERVICE) and not mpi_comm.iprobe(source=reader_rank, tag=MPI_TAGS.DATA):
        #     if mpi_comm.recv(source=reader_rank, tag=MPI_TAGS.SERVICE) == 1:
        #         break

    mpi_comm.send(obj=1, dest=trt_rank, tag=MPI_TAGS.SERVICE)
    mpi_comm.send(obj=STATE.EXITED, dest=0, tag=MPI_TAGS.STATE)
    sts.logger.info("Exiting...")
    return 0


if __name__ == "__main__":
    pass
