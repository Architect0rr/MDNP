#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 10-09-2023 07:26:43

import csv
import time
import json
from pathlib import Path
from typing import List, Dict, Union, Any

import adios2
import numpy as np

from ...utils import Role
from .... import constants as cs
from .utils import distribute
from ...utils_mpi import MC, MPI_TAGS


def gen_matrix(cwd: Path, params: Dict, storages: List[Path], cut: int):
    output_csv_fp = cwd / params[cs.fields.data_processing_folder] / cs.files.cluster_distribution_matrix
    with open(output_csv_fp, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for storage in storages:
            with adios2.open(storage.as_posix(), 'r') as reader:  # type: ignore
                for step in reader:
                    stee: int = step.read(cs.lcf.mat_step)
                    dist = step.read(cs.lcf.mat_dist)
                    writer.writerow(np.hstack([stee, dist[:cut]]).astype(dtype=np.uint32).flatten())


def after_new(sts: MC, m: int):
    cwd, mpi_comm, mpi_rank, mpi_size = sts.cwd, sts.mpi_comm, sts.mpi_rank, sts.mpi_size
    mpi_comm.Barrier()
    print(f"MPI rank {mpi_rank}, barrier off")
    states = {}

    response_array = []
    while True:
        for i in range(1, mpi_size):
            if mpi_comm.iprobe(source=i, tag=MPI_TAGS.ONLINE):
                resp = mpi_comm.recv(source=i, tag=MPI_TAGS.ONLINE)
                print(f"Recieved from {i}: {resp}")
                states[str(i)] = {}
                states[str(i)][cs.cf.pp_state_name] = resp
                response_array.append((i, resp))
        if len(response_array) == mpi_size - 1:
            break
    mpi_comm.Barrier()
    print(f"MPI rank {mpi_rank}, second barrier off")
    completed_threads = []
    fl = True
    start = time.time()
    while fl:
        for i in range(m, mpi_size):
            if mpi_comm.iprobe(source=i, tag=MPI_TAGS.STATE):
                tstate: int = mpi_comm.recv(source=i, tag=MPI_TAGS.STATE)
                if tstate == -1:
                    completed_threads.append(i)
                    print(f"MPI ROOT, rank {i} has been completed")
                    if len(completed_threads) == mpi_size - m:
                        with open(cwd / cs.files.post_process_state, "w") as fp:
                            json.dump(states, fp)
                        fl = False
                        break
                else:
                    states[str(i)][cs.cf.pp_state_name] = tstate
        if time.time() - start > 20:
            with open(cwd / cs.files.post_process_state, "w") as fp:
                json.dump(states, fp)
            start = time.time()
    for i in range(1, m):
        mpi_comm.send(obj=-1, dest=i, tag=MPI_TAGS.COMMAND)

    storages = []
    max_sizes = []
    for i in range(m, mpi_size):
        storage: Path
        max_cluster_size: int
        storage, max_cluster_size = mpi_comm.recv(source=i, tag=MPI_TAGS.SERV_DATA_3)
        storages.append((i, storage))
        max_sizes.append(max_cluster_size)

    # storages = [(i, storage) for i, storage in enumerate(storages)]
    storages.sort(key=lambda x: x[1])
    storages = [storage[1] for storage in storages]

    data_file = (cwd / cs.files.data)
    with open(data_file, 'r') as fp:
        son: Dict[str, Any] = json.load(fp)

    son[cs.cf.matrix_storages] = [storage.as_posix() for storage in storages]

    with open(data_file, 'w') as fp:
        json.dump(son, fp)

    gen_matrix(cwd, son, storages, max(max_sizes))

    print("MPI ROOT: exiting...")
    return 0


def new(sts: MC, params: Dict, nv: int):
    mpi_comm, mpi_size = sts.mpi_comm, sts.mpi_size

    thread_num = mpi_size - nv
    print(f"Thread num: {thread_num}")
    wd: Dict[str, Dict[str, Union[int, Dict[str, int]]]] = distribute(params[cs.fields.storages], thread_num)
    print("Distribution")
    print(json.dumps(wd, indent=4))

    for i in range(thread_num):
        mpi_comm.send(obj=Role.matr, dest=i + nv, tag=MPI_TAGS.DISTRIBUTION)

    for i in range(thread_num):
        mkl = (wd[str(i)][cs.fields.number], wd[str(i)][cs.fields.storages])
        mpi_comm.send(obj=mkl, dest=nv + i, tag=MPI_TAGS.SERV_DATA_1)
        mpi_comm.send(obj=params, dest=nv + i, tag=MPI_TAGS.SERV_DATA_2)

    return after_new(sts, 1)
