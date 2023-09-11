#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 11-09-2023 18:06:15


import os
from typing import Literal, Dict, Any


os.environ['OPENBLAS_NUM_THREADS'] = '1'


import freud  # type: ignore
import numpy as np
from numpy import typing as npt
import pandas as pd  # type: ignore

from ....core import calc
from ...utils import STATE
from .... import constants as cs
from ...utils_mpi import MC, MPI_TAGS


def treat_mpi(sts: MC) -> Literal[0]:
    cwd, mpi_comm, mpi_rank = sts.cwd, sts.mpi_comm, sts.mpi_rank

    sts.logger.info("Receiving data from root")
    params: Dict[str, Any] = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA)
    sts.logger.info("Data received")

    N_atoms: int = params[cs.fields.N_atoms]
    bdims: npt.NDArray[np.float32] = params[cs.fields.dimensions]
    dt: float = params[cs.fields.time_step]
    dis: int = params[cs.fields.every]

    proc_rank = mpi_rank - 1

    box = freud.box.Box.from_box(np.array(bdims))
    volume = box.volume
    sizes: npt.NDArray[np.uint32] = np.arange(1, N_atoms + 1, dtype=np.uint64)

    sts.logger.info("Trying to read temperature file")
    temperatures = pd.read_csv(cwd / cs.files.temperature, header=None)
    temptime = temperatures[0].to_numpy(dtype=np.uint64)
    temperatures = temperatures[1].to_numpy(dtype=np.float64)

    sts.logger.info("Stating main loop")
    while not mpi_comm.iprobe(source=proc_rank, tag=MPI_TAGS.SERVICE) or mpi_comm.iprobe(source=proc_rank, tag=MPI_TAGS.DATA):
        step: int
        dist: npt.NDArray[np.uint32]
        step, dist = mpi_comm.recv(source=proc_rank, tag=MPI_TAGS.DATA)

        try:
            km = 10
            temp = temperatures[np.abs(temptime - int(step * dis)) <= 1][0]  # type: ignore
            tow = calc.get_row(step, sizes, dist, temp, N_atoms, volume, dt, dis, km)
        except Exception as e:
            sts.logger.error(f"Exception at step: {step}")
            sts.logger.exception(e)
            sts.logger.error("Writing zeroes")
            tow = np.zeros(10, dtype=np.float32)

        mpi_comm.send(obj=tow, dest=1, tag=MPI_TAGS.WRITE)
        mpi_comm.send(obj=step, dest=0, tag=MPI_TAGS.STATE)

        # if mpi_comm.iprobe(source=proc_rank, tag=MPI_TAGS.SERVICE) and not mpi_comm.iprobe(source=proc_rank, tag=MPI_TAGS.DATA):
        #     if mpi_comm.recv(source=proc_rank, tag=MPI_TAGS.SERVICE) == 1:
        #         break

    mpi_comm.send(obj=STATE.EXITED, dest=0, tag=MPI_TAGS.STATE)
    sts.logger.info("Exiting...")
    return 0


if __name__ == "__main__":
    pass
