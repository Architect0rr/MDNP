#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 12-09-2023 10:31:58

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Literal

import adios2  # type: ignore
import numpy as np
from numpy import typing as npt

from .. import constants as cs
# from ..core.distribution import distribution


def adw(adout, name, arr, end=False):
    if end:
        adout.write(name, arr, arr.shape, np.full(len(arr.shape), 0), arr.shape, end_step=True)  # type: ignore
    else:
        adout.write(name, arr, arr.shape, np.full(len(arr.shape), 0), arr.shape)  # type: ignore


def proceed(cwd: Path, storages: List[str], process_folder: str, Natoms: int, logger: logging.Logger):
    ndim = 3
    kB = 1.380649e-23
    worker_counter = 0
    max_cluster_size: int = 0
    sizes: npt.NDArray[np.uint64] = np.arange(1, Natoms + 1, dtype=np.uint64)
    ntb_fp: Path = cwd / process_folder / "ntb.bp"
    logger.info(f"Trying to create adios storage: {ntb_fp.as_posix()}")
    with adios2.open(ntb_fp.as_posix(), 'w') as adout:  # type: ignore
        logger.info("Stating main loop")
        for storage in storages:
            storage_fp = (cwd / storage).as_posix()
            logger.debug(f"Trying to open {storage_fp}")
            with adios2.open(storage_fp, 'r') as reader:  # type: ignore
                i = 0
            for fstep in reader:
                arr = fstep.read(cs.lcf.lammps_dist)
                real_timestep = fstep.read(cs.lcf.real_timestep)
                ids = arr[:, 0].astype(dtype=np.uint64)
                cl_ids = arr[:, 1].astype(dtype=np.uint64)
                masses = arr[:, 2].astype(dtype=np.uint64)
                vxs = arr[:, 3].astype(dtype=np.float32)
                vys = arr[:, 4].astype(dtype=np.float32)
                vzs = arr[:, 5].astype(dtype=np.float32)
                # temp = arr[:, 6].astype(dtype=np.float32)

                cl_unique_ids = np.unique(cl_ids)
                ids_by_cl_id = [ids[cl_ids == i] for i in cl_unique_ids]
                cl_sizes = np.array([len(ids) for ids in ids_by_cl_id])
                cl_unique_sizes, sizes_cnt = np.unique(cl_sizes, return_counts=True)

                dist = np.zeros(Natoms + 1, dtype=np.uint32)
                dist[cl_unique_sizes] = sizes_cnt
                dist = dist[1:]

                cl_ids_by_size = [cl_unique_ids[cl_sizes == i] for i in cl_unique_sizes]
                ids_by_size = [np.hstack([ids_by_cl_id[cl_unique_ids == cl_ids_ws_wid][0] for cl_ids_ws_wid in cl_ids_ws]) for cl_ids_ws in cl_ids_by_size]

                vs_square = vxs**2 + vys**2 + vzs**2
                kes = masses * vs_square / 2
                sum_ke_by_size = [np.sum(np.hstack([kes[ids == id] for id in ids_s])) for ids_s in ids_by_size]
                atom_counts_by_size = cl_unique_sizes*sizes_cnt
                ndofs_by_size = atom_counts_by_size*(ndim-1)
                temp_by_size = (sum_ke_by_size / ndofs_by_size) * 2 / kB

                # stepnd = worker_counter + ino

                adout.write(cs.lcf.real_timestep, real_timestep)  # type: ignore
                adout.write(cs.lcf.worker_step, np.array(worker_counter))  # type: ignore
                adw(adout, cs.lcf.sizes, cl_unique_sizes)
                adw(adout, cs.lcf.size_counts, sizes_cnt)
                adw(adout, cs.lcf.cl_temps, temp_by_size)
                adw(adout, cs.lcf.mat_dist, dist, True)

                max_cluster_size = int(np.argmax(sizes[dist != 0]) + 1)

                worker_counter += 1

                # if i == storages[storage][cs.fields.end] + storages[storage][cs.fields.begin] - 1:
                #     print(f"Reached end of distribution, {storage, i, worker_counter}")
                #     break

                i += 1

    logger.info("Done")
    logger.info("Creating csv info matrix")

    return 0


def setup_logger(cwd: Path, name: str, level: int = logging.INFO) -> logging.Logger:
    folder = cwd / cs.folders.log
    folder.mkdir(exist_ok=True, parents=True)
    folder = folder / cs.folders.post_process_log
    folder.mkdir(exist_ok=True, parents=True)
    logfile = folder / "simple.log"

    handler = logging.FileHandler(logfile)
    handler.setFormatter(cs.obj.formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def main() -> Literal[0]:
    cwd = Path.cwd()

    parser = argparse.ArgumentParser(description='Generate cluster distribution matrix from ADIOS2 LAMMPS data.')
    parser.add_argument('--debug', action='store_true', help='Debug, prints only parsed arguments')
    parser.add_argument('--mode', action='store', type=int, default=3, help='Mode to run')
    args = parser.parse_args()

    logger = setup_logger(cwd, 'simple', logging.DEBUG)

    data_file = (cwd / cs.files.data)
    logger.debug("Reading datafile")
    with open(data_file, 'r') as fp:
        params: Dict[str, Any] = json.load(fp)

    storages: List[str] = params[cs.fields.storages]
    data_folder = params[cs.fields.data_processing_folder]
    Natoms = params[cs.fields.N_atoms]

    return proceed(cwd, storages, data_folder, Natoms, logger.getChild('proc'))


if __name__ == "__main__":
    import sys
    sys.exit(main())