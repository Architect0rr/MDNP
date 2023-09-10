#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 10-09-2023 07:22:06

import time
import json
from typing import Dict, Union

import numpy as np

from .... import constants as cs
from ...utils_mpi import MC, MPI_TAGS


def after_ditribution(sts: MC, m: int):
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
                tstate = mpi_comm.recv(source=i, tag=MPI_TAGS.STATE)
                if tstate == -1:
                    completed_threads.append(i)
                    print(f"MPI ROOT, rank {i} has completed")
                    if len(completed_threads) == mpi_size - m:
                        with open(cwd / cs.files.post_process_state, "w") as fp:
                            json.dump(states, fp)
                        fl = False
                        break
                else:
                    states[str(i)][cs.cf.pp_state] = tstate
        if time.time() - start > 20:
            with open(cwd / cs.files.post_process_state, "w") as fp:
                json.dump(states, fp)
            start = time.time()
    for i in range(1, m):
        mpi_comm.send(obj=-1, dest=i, tag=MPI_TAGS.COMMAND)
    print("MPI ROOT: exiting...")

    return 0


def distribute(storages: Dict[str, int], mm: int) -> Dict[str, Dict[str, Union[int, Dict[str, int]]]]:
    ll = sum(list(storages.values()))
    dp = np.linspace(0, ll - 1, mm + 1, dtype=int)
    bp = dp
    dp = dp[1:] - dp[:-1]
    dp = np.vstack([bp[:-1].astype(dtype=int), np.cumsum(dp).astype(dtype=int)])
    wd = {}
    st = {}
    for storage, value in storages.items():
        st[storage] = value
    ls = 0
    for i, (begin_, end_) in enumerate(dp.T):
        begin = int(begin_)
        end = int(end_)
        beg = 0 + ls
        en = end - begin
        wd[str(i)] = {cs.fields.number: begin, cs.fields.storages: {}}
        for storage in list(st):
            value = st[storage]
            if en >= value:
                wd[str(i)][cs.fields.storages][storage] = {cs.fields.begin: beg, cs.fields.end: value}
                en -= value
                ls = 0
                beg = 0
                del st[storage]
            elif en < value:
                wd[str(i)][cs.fields.storages][storage] = {cs.fields.begin: beg, cs.fields.end: en}
                st[storage] -= en
                ls += en
                break
    return wd
