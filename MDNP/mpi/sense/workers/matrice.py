#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 12-09-2023 01:50:54

from typing import Dict
from pathlib import Path

import freud  # type: ignore
import adios2  # type: ignore
import numpy as np
from numpy import typing as npt

from ...utils import STATE
from ...utils_mpi import MC, MPI_TAGS
from .... import constants as cs
from ....core import distribution


def thread(sts: MC):
    cwd: Path
    cwd, mpi_comm, mpi_rank = sts.cwd, sts.mpi_comm, sts.mpi_rank

    sts.logger.info("Receiving storages")
    ino: int
    storages: Dict[str, int]
    ino, storages = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA_1)
    sts.logger.info("Storages received")

    sts.logger.info("Receiving paramseters")
    params = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA_2)
    sts.logger.info("Parameters received")

    N_atoms: int = params[cs.fields.N_atoms]
    bdims: npt.NDArray[np.float32] = params[cs.fields.dimensions]
    box = freud.box.Box.from_box(bdims)
    sizes = np.arange(1, N_atoms + 1, 1)

    max_cluster_size = 0
    worker_counter = 0
    ntb_fp: Path = cwd / params[cs.fields.data_processing_folder] / f"ntb.{mpi_rank}.bp"
    sts.logger.info(f"Trying to create adios storage: {ntb_fp.as_posix()}")
    with adios2.open(ntb_fp.as_posix(), 'w') as adout:  # type: ignore
        sts.logger.info("Stating main loop")
        storage: str
        for storage in storages:
            storage_fp = (cwd / storage).as_posix()
            with adios2.open(storage_fp, 'r') as reader:  # type: ignore
                i = 0
                for step in reader:
                    if i < storages[storage][cs.fields.begin]:  # type: ignore
                        i += 1
                        continue
                    arr = step.read(cs.lcf.lammps_dist)
                    arr = arr[:, 2:5].astype(dtype=np.float32)

                    stepnd = worker_counter + ino

                    dist = distribution.get_dist(arr, N_atoms, box)

                    adout.write(cs.lcf.mat_step, np.array(stepnd))  # type: ignore
                    adout.write(cs.lcf.mat_dist, dist, dist.shape, np.full(len(dist.shape), 0), dist.shape, end_step=True)  # type: ignore

                    max_cluster_size = int(np.argmax(sizes[dist != 0]))

                    worker_counter += 1
                    mpi_comm.send(obj=worker_counter, dest=0, tag=MPI_TAGS.STATE)

                    if i == storages[storage][cs.fields.end] + storages[storage][cs.fields.begin] - 1:  # type: ignore
                        break

                    i += 1

    sts.logger.info("Reached end")
    mpi_comm.send(obj=STATE.EXITED, dest=0, tag=MPI_TAGS.STATE)
    mpi_comm.send(obj=(ntb_fp, max_cluster_size), dest=0, tag=MPI_TAGS.SERV_DATA_3)
    sts.logger.info("Exiting...")
    return 0


if __name__ == "__main__":
    pass
