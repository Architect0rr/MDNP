#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 18-09-2023 12:15:33

import time
# import json
from typing import Tuple, List, Dict, Union

import numpy as np

from ...utils import Role
from .... import constants as cs
from ...utils_mpi import MC, MPI_TAGS
from ...utils import COMMAND, STATE


def gw2c(sts: MC, nv: int):  # gather, wait to complete
    sts.logger = sts.logger.getChild('gw2c')

    sts.logger.info("Releasing distribution barrier")
    sts.mpi_comm.Barrier()
    sts.logger.info("Distribution barrier released")

    sts.logger.info("Gathering information ")
    response_array: List[Tuple[int, Role]] = []
    while len(response_array) < sts.mpi_size - 1:
        # sts.logger.debug("Gathering information ")
        for i in range(1, sts.mpi_size):
            if sts.mpi_comm.iprobe(source=i, tag=MPI_TAGS.ONLINE):
                resp = sts.mpi_comm.recv(source=i, tag=MPI_TAGS.ONLINE)
                sts.logger.debug(f"Received from {i}: {resp}")
                response_array.append((i, resp))

    sts.logger.info("Releasing second barrier")
    sts.mpi_comm.Barrier()
    sts.logger.info("Second barrier released")

    # states = {}

    completed_threads: List[int] = []
    # fl = True
    # start = time.time()
    sts.logger.info("Starting main loop, waiting for workers to complete with 20 sec sleep")
    while len(completed_threads) < sts.mpi_size - nv:
        for i in range(nv, sts.mpi_size):
            if sts.mpi_comm.iprobe(source=i, tag=MPI_TAGS.STATE):
                tstate = sts.mpi_comm.recv(source=i, tag=MPI_TAGS.STATE)
                if tstate == STATE.EXITED:
                    completed_threads.append(i)
                    sts.logger.info(f"Rank {i} has been completed")
                    # if len(completed_threads) == sts.mpi_size - m:
                    #     with open(sts.cwd / cs.files.post_process_state, "w") as fp:
                    #         json.dump(states, fp)
                    #     fl = False
                    #     break
                elif tstate == STATE.EXCEPTION:
                    sts.logger.critical(f"Uncaught exception in rank: {i}, trying to stop all, exiting...")
                    for des in range(1, sts.mpi_size):
                        sts.mpi_comm.send(obj=COMMAND.EXIT, dest=des, tag=MPI_TAGS.COMMAND)
                    raise RuntimeError(f"Uncaught exception in rank: {i}, trying to stop all, exiting...")
                # else:
                #     states[str(i)][cs.cf.pp_state] = tstate
        if not any([sts.mpi_comm.iprobe(source=i, tag=MPI_TAGS.STATE) for i in range(nv, sts.mpi_size)]):
            time.sleep(20)
        # if time.time() - start > 20:
        #     with open(sts.cwd / cs.files.post_process_state, "w") as fp:
        #         json.dump(states, fp)
        #     start = time.time()

    sts.logger.info("All workers exited")
    sts.logger.info("Sending exit command to all ranks")
    for i in range(1, nv):
        sts.mpi_comm.send(obj=COMMAND.EXIT, dest=i, tag=MPI_TAGS.COMMAND)


def after_ditribution(sts: MC, nv: int):

    gw2c(sts, nv)

    sts.logger.info("Exiting...")

    return 0


def distribute(storages: Dict[str, int], mm: int) -> Dict[str, Dict[str, Union[int, Dict[str, int]]]]:
    ll = sum(list(storages.values()))
    dp = np.linspace(0, ll - 1, mm + 1, dtype=int)
    bp = dp
    dp = dp[1:] - dp[:-1]
    dp = np.vstack([bp[:-1].astype(dtype=int), np.cumsum(dp).astype(dtype=int)])
    wd: Dict[str, Dict[str, Union[int, Dict[str, int]]]] = {}
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
                wd[str(i)][cs.fields.storages][storage] = {cs.fields.begin: beg, cs.fields.end: value}  # type: ignore
                en -= value
                ls = 0
                beg = 0
                del st[storage]
            elif en < value:
                wd[str(i)][cs.fields.storages][storage] = {cs.fields.begin: beg, cs.fields.end: en}  # type: ignore
                st[storage] -= en
                ls += en
                break
    return wd
