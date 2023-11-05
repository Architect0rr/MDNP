#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 22-09-2023 03:19:22

# import argparse
from pathlib import Path
from typing import Dict
from venv import logger

import adios2  # type: ignore
import numpy as np
from numpy import typing as npt

from ...utils import STATE
from .... import constants as cs
from ...utils_mpi import MC, MPI_TAGS
from ...adios_wrap import adser


def adw(adout, name, arr, end=False):
    if end:
        adout.write(name, arr, arr.shape, np.full(len(arr.shape), 0), arr.shape, end_step=True)  # type: ignore
    else:
        adout.write(name, arr, arr.shape, np.full(len(arr.shape), 0), arr.shape)  # type: ignore


def simple(sts: MC):
    sts.logger.info("Receiving storages")
    ino: int
    storages: Dict[str, Dict[str, int]]
    ino, storages = sts.mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA_1)
    sts.logger.info("Storages received")

    sts.logger.info("Receiving paramseters")
    params = sts.mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA_2)
    sts.logger.info("Parameters received")

    Natoms: int = params[cs.fields.N_atoms]
    sizes: npt.NDArray[np.uint64] = np.arange(1, Natoms + 1, dtype=np.uint64)

    sts.logger.info("Setting up ADIOS2 output")
    adout = adser(sts, mpi=False)
    sts.logger.debug("Declaring variables")
    adout.declare_arr(cs.lcf.worker_step, 1, np.int64)
    adout.declare_arr(cs.lcf.mat_step, 1, np.int64)
    adout.declare_arr(cs.lcf.real_timestep, 1, np.int64)
    adout.declare_arr(cs.lcf.tot_temp, 1, np.float32)

    adout.declare_arr(cs.lcf.sizes, Natoms, np.int64)
    adout.declare_arr(cs.lcf.size_counts, Natoms, np.int64)
    adout.declare_arr(cs.lcf.mat_dist, Natoms, np.int64)
    adout.declare_arr(cs.lcf.cl_temps, Natoms, np.float64)

    ndim = 3
    worker_counter = 0
    max_cluster_size: int = 0
    ntb_fp: Path = sts.cwd / params[cs.fields.data_processing_folder] / f"ntb.{sts.mpi_rank}.bp"
    sts.logger.info(f"Trying to create adios storage: {ntb_fp.as_posix()}")
    adout.open(ntb_fp.as_posix())
    sts.logger.info("Stating main loop")
    for storage in storages.keys():
        storage_fp = (sts.cwd / storage).as_posix()
        sts.logger.debug(f"Trying to open {storage_fp}")
        with adios2.open(storage_fp, 'r') as reader:  # type: ignore
            sts.logger.debug("Started this storage")
            i = 0
            for fstep in reader:
                if i < storages[storage][cs.fields.begin]:
                    i += 1
                    continue
                stepnd = worker_counter + ino

                arr = fstep.read(cs.lcf.lammps_dist)
                arr = arr[arr[:, 0].argsort()]
                real_timestep = fstep.read(cs.lcf.real_timestep)
                ids = arr[:, 0].astype(dtype=np.int64)
                cl_ids = arr[:, 1].astype(dtype=np.int64)
                masses = arr[:, 2].astype(dtype=np.int64)
                vxs = arr[:, 3].astype(dtype=np.float32)
                vys = arr[:, 4].astype(dtype=np.float32)
                vzs = arr[:, 5].astype(dtype=np.float32)
                # temp = arr[:, 6].astype(dtype=np.float32)

                together = np.vstack([cl_ids, ids]).T
                together = together[together[:, 0].argsort()]
                ids_by_cl_id = np.split(together[:, 1], np.unique(together[:, 0], return_index=True)[1][1:])

                cl_unique_ids = np.arange(1, len(ids_by_cl_id)+1, dtype=np.int64)

                cl_sizes = np.array([len(ids) for ids in ids_by_cl_id])
                cl_unique_sizes, sizes_cnt = np.unique(cl_sizes, return_counts=True)

                dist = np.zeros(Natoms + 1, dtype=np.uint32)
                dist[cl_unique_sizes] = sizes_cnt
                dist = dist[1:]

                cl_unique_sizes.sort()

                ids_n_sizes = np.vstack([cl_sizes, cl_unique_ids]).T
                ids_n_sizes = ids_n_sizes[ids_n_sizes[:, 0].argsort()]
                cl_ids_by_size = np.split(ids_n_sizes[:, 1], np.unique(ids_n_sizes[:, 0], return_index=True)[1][1:])

                ids_by_cl_id = np.array(ids_by_cl_id, dtype=object)
                ids_by_size = [np.stack(np.take(ids_by_cl_id, (cl_ids_w_size-1))).flatten('A') for cl_ids_w_size in cl_ids_by_size]  # type: ignore

                vs_square = vxs**2 + vys**2 + vzs**2
                kes = masses * vs_square / 2
                sum_ke_by_size = np.array([np.sum(np.take(kes, ids_s-1)) for ids_s in ids_by_size])
                atom_counts_by_size = cl_unique_sizes*sizes_cnt
                ndofs_by_size = (atom_counts_by_size-1)*ndim
                try:
                    temp_by_size = (sum_ke_by_size / ndofs_by_size) * 2
                except Exception:
                    logger.exception("Pass... assuming zero")
                    temp_by_size = np.zeros_like(sum_ke_by_size)

                total_temp = (np.sum(kes) / ((Natoms - 1) * ndim)) * 2

                adout.begin_step()
                adout.wr_array(cs.lcf.worker_step, np.array(worker_counter))
                adout.wr_array(cs.lcf.tot_temp, np.array(total_temp))
                adout.wr_array(cs.lcf.mat_step, np.array(stepnd))
                adout.wr_array(cs.lcf.real_timestep, np.array(real_timestep))
                adout.wr_array(cs.lcf.sizes, cl_unique_sizes)
                adout.wr_array(cs.lcf.size_counts, sizes_cnt)
                adout.wr_array(cs.lcf.cl_temps, temp_by_size)
                adout.wr_array(cs.lcf.mat_dist, dist)
                adout.end_step()

                if max_cluster_size < int(np.argmax(sizes[dist != 0]) + 1):
                    max_cluster_size = int(np.argmax(sizes[dist != 0]) + 1)

                worker_counter += 1
                sts.mpi_comm.send(obj=worker_counter, dest=0, tag=MPI_TAGS.STATE)

                if i == storages[storage][cs.fields.end] + storages[storage][cs.fields.begin] - 1:  # type: ignore
                    break

                i += 1

    adout.close()

    sts.logger.info("Reached end")
    sts.mpi_comm.send(obj=STATE.EXITED, dest=0, tag=MPI_TAGS.STATE)
    sts.mpi_comm.send(obj=(ntb_fp, max_cluster_size), dest=0, tag=MPI_TAGS.SERV_DATA_3)
    sts.logger.info("Exiting...")
    return 0


if __name__ == "__main__":
    pass
