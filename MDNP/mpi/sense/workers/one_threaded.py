#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 11-09-2023 20:29:51

import csv
from typing import Dict

import freud   # type: ignore
import adios2  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from numpy import typing as npt

from ...utils import STATE
from .... import constants as cs
from ...utils_mpi import MC, MPI_TAGS
from ....core import distribution, calc


def thread(sts: MC):
    cwd, mpi_comm, mpi_rank = sts.cwd, sts.mpi_comm, sts.mpi_rank

    sts.logger.info("Receiving storages")
    ino: int
    storages: Dict[str, int]
    ino, storages = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA_1)
    sts.logger.info("Storages received")

    sts.logger.info("Receiving parameters")
    params = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA_2)
    sts.logger.info("Parameters received")
    N_atoms: int = params[cs.fields.N_atoms]
    bdims: npt.NDArray[np.float32] = params[cs.fields.dimensions]
    dt: float = params[cs.fields.time_step]
    dis: int = params[cs.fields.every]

    box = freud.box.Box.from_box(bdims)
    volume = box.volume
    sizes: npt.NDArray[np.uint32] = np.arange(1, N_atoms + 1, dtype=np.uint32)

    sts.logger.info("Trying to read temperature file")
    temperatures_mat = pd.read_csv(cwd / cs.files.temperature, header=None)
    temptime = temperatures_mat[0].to_numpy(dtype=np.uint64)
    temperatures = temperatures_mat[1].to_numpy(dtype=np.float64)

    worker_counter = 0
    output_csv_fp = (cwd / params[cs.fields.data_processing_folder] / f"rdata.{mpi_rank}.csv").as_posix()
    ntb_fp = (cwd / params[cs.fields.data_processing_folder] / f"ntb.{mpi_rank}.bp").as_posix()
    sts.logger.info(f"Trying to create adios storage: {ntb_fp}")
    sts.logger.info(f"Trying to open csv file: {ntb_fp}")
    with adios2.open(ntb_fp, 'w') as adout, open(output_csv_fp, "w") as csv_file:  # type: ignore
        writer = csv.writer(csv_file, delimiter=',')
        storage: str
        sts.logger.info("Stating main loop")
        for storage in storages:
            storage_fp = (cwd / storage).as_posix()
            with adios2.open(storage_fp, 'r') as reader:  # type: ignore
                total_steps = reader.steps()
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

                    km = 10
                    temp = temperatures[np.abs(temptime - int(stepnd * dis)) <= 1][0]
                    tow = calc.get_row(stepnd, sizes, dist, temp, N_atoms, volume, dt, dis, km)

                    writer.writerow(tow)
                    csv_file.flush()

                    worker_counter += 1
                    mpi_comm.send(obj=worker_counter, dest=0, tag=MPI_TAGS.STATE)

                    if i == storages[storage][cs.fields.end] + storages[storage][cs.fields.begin] - 1:  # type: ignore
                        sts.logger.info("Reached end of storage by soft stop")
                        break

                    i += 1

                    if step.current_step() == total_steps - 1:
                        sts.logger.info("Reached end of storage by hard stop")
                        break

    sts.logger.info("Reached end")
    mpi_comm.send(obj=STATE.EXITED, dest=0, tag=MPI_TAGS.STATE)
    sts.logger.info("Exiting...")
    return 0


if __name__ == "__main__":
    pass
