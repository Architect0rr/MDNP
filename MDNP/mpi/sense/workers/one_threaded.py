#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 10-09-2023 08:09:16

import csv
from typing import Dict

import freud
import adios2
import numpy as np
import pandas as pd
from numpy import typing as npt

from .... import constants as cs
from ...utils_mpi import MC, MPI_TAGS
from ....core import distribution, calc


def thread(sts: MC):
    cwd, mpi_comm, mpi_rank = sts.cwd, sts.mpi_comm, sts.mpi_rank
    mpi_comm.Barrier()

    ino: int
    storages: Dict[str, int]
    ino, storages = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA_1)

    params = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA_2)
    N_atoms: int = params[cs.fields.N_atoms]
    bdims: npt.NDArray[np.float32] = params[cs.fields.dimensions]
    dt: float = params[cs.fields.time_step]
    dis: int = params[cs.fields.every]

    box = freud.box.Box.from_box(bdims)
    volume = box.volume
    sizes: npt.NDArray[np.uint32] = np.arange(1, N_atoms + 1, dtype=np.uint32)

    temperatures_mat = pd.read_csv(cwd / cs.files.temperature, header=None)
    temptime = temperatures_mat[0].to_numpy(dtype=np.uint64)
    temperatures = temperatures_mat[1].to_numpy(dtype=np.float64)

    worker_counter = 0
    print(f"MPI rank {mpi_rank}, reader, storages: {storages}")
    output_csv_fp = (cwd / params[cs.fields.data_processing_folder] / f"rdata.{mpi_rank}.csv").as_posix()
    ntb_fp = (cwd / params[cs.fields.data_processing_folder] / f"ntb.{mpi_rank}.bp").as_posix()
    with adios2.open(ntb_fp, 'w') as adout, open(output_csv_fp, "w") as csv_file:  # type: ignore
        writer = csv.writer(csv_file, delimiter=',')
        storage: str
        for storage in storages:
            storage_fp = (cwd / storage).as_posix()
            with adios2.open(storage_fp, 'r') as reader:  # type: ignore
                total_steps = reader.steps()
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

                    km = 10
                    temp = temperatures[np.abs(temptime - int(stepnd * dis)) <= 1][0]
                    tow = calc.get_row(stepnd, sizes, dist, temp, N_atoms, volume, dt, dis, km)

                    writer.writerow(tow)
                    csv_file.flush()

                    worker_counter += 1
                    mpi_comm.send(obj=worker_counter, dest=0, tag=MPI_TAGS.STATE)

                    if i == storages[storage][cs.cf.end] + storages[storage][cs.cf.begin] - 1:  # type: ignore
                        print(f"MPI rank {mpi_rank}, reader, reached end of distribution, {storage, i, worker_counter}")
                        break

                    i += 1

                    if step.current_step() == total_steps - 1:
                        print(f"MPI rank {mpi_rank}, reader, reached end of storage, {storage, i, worker_counter}")
                        break

    mpi_comm.send(obj=-1, dest=0, tag=MPI_TAGS.STATE)


if __name__ == "__main__":
    pass
