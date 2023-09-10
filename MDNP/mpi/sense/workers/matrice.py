#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 10-09-2023 08:08:04

from typing import Dict
from pathlib import Path

import freud
import adios2
import numpy as np
from numpy import typing as npt

from ...utils_mpi import MC, MPI_TAGS
from .... import constants as cs
from ....core import distribution


def thread(sts: MC):
    cwd: Path
    cwd, mpi_comm, mpi_rank = sts.cwd, sts.mpi_comm, sts.mpi_rank
    mpi_comm.Barrier()

    ino: int
    storages: Dict[str, int]
    ino, storages = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA_1)

    params = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA_2)

    N_atoms: int = params[cs.fields.N_atoms]
    bdims: npt.NDArray[np.float32] = params[cs.fields.dimensions]
    box = freud.box.Box.from_box(bdims)
    sizes = np.arange(1, N_atoms + 1, 1)

    max_cluster_size = 0
    worker_counter = 0
    print(f"MPI rank {mpi_rank}, reader, storages: {storages}")
    ntb_fp: Path = cwd / params[cs.fields.data_processing_folder] / f"ntb.{mpi_rank}.bp"
    with adios2.open(ntb_fp.as_posix(), 'w') as adout:  # type: ignore
        storage: str
        for storage in storages:
            storage_fp = (cwd / storage).as_posix()
            with adios2.open(storage_fp, 'r') as reader:  # type: ignore
                i = 0
                for step in reader:
                    if i < storages[storage][cs.cf.begin]:  # type: ignore
                        i += 1
                        continue
                    arr = step.read(cs.lcf.lammps_dist)
                    arr = arr[:, 2:5].astype(dtype=np.float32)

                    stepnd = worker_counter + ino

                    dist = distribution.get_dist(arr, N_atoms, box)

                    adout.write(cs.cf.mat_step, np.array(stepnd))  # type: ignore
                    adout.write(cs.cf.mat_dist, dist, dist.shape, np.full(len(dist.shape), 0), dist.shape, end_step=True)  # type: ignore

                    max_cluster_size = np.argmax(sizes[dist != 0])

                    worker_counter += 1
                    mpi_comm.send(obj=worker_counter, dest=0, tag=MPI_TAGS.STATE)

                    if i == storages[storage][cs.cf.end] + storages[storage][cs.cf.begin] - 1:  # type: ignore
                        print(f"MPI rank {mpi_rank}, reader, reached end of distribution, {storage, i, worker_counter}")
                        break

                    i += 1

    mpi_comm.send(obj=-1, dest=0, tag=MPI_TAGS.STATE)
    mpi_comm.send(obj=(ntb_fp, max_cluster_size), dest=0, tag=MPI_TAGS.SERV_DATA_3)


if __name__ == "__main__":
    pass
