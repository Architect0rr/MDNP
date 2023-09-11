#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 11-09-2023 20:26:46


import time
from typing import Dict, Literal, Union

import adios2  # type: ignore
import numpy as np

from ...utils import STATE
from .... import constants as cs
from ...utils_mpi import MC, MPI_TAGS


def reader(sts: MC) -> Literal[0]:
    cwd, mpi_comm, mpi_rank = sts.cwd, sts.mpi_comm, sts.mpi_rank

    sts.logger.info("Receiving storages")
    dasdictt: Dict[str, Union[int, Dict[str, int]]] = mpi_comm.recv(source=0, tag=MPI_TAGS.SERV_DATA)
    sts.logger.info("Storages received")
    ino: int = dasdictt[cs.fields.number]  # type: ignore
    storages: Dict[str, int] = dasdictt[cs.fields.storages]  # type: ignore

    proceeder_rank = mpi_rank + 1
    worker_counter = 0
    sync_value: int = 0

    sts.logger.info("Started main loop")
    storage: str
    for storage in storages:
        with adios2.open((cwd / storage).as_posix(), 'r') as reader:  # type: ignore
            total_steps = reader.steps()
            i = 0
            for step in reader:
                if i < storages[storage][cs.fields.begin]:  # type: ignore
                    i += 1
                    continue
                arr = step.read(cs.lcf.lammps_dist)
                arr = arr[:, 2:5].astype(dtype=np.float32)
                tpl = (worker_counter + ino, mpi_rank, arr)
                # print(f"MPI rank {mpi_rank}, reader, {worker_counter}")

                mpi_comm.send(obj=tpl, dest=proceeder_rank, tag=MPI_TAGS.DATA)
                worker_counter += 1
                mpi_comm.send(obj=worker_counter, dest=0, tag=MPI_TAGS.STATE)

                if i >= storages[storage][cs.fields.end] + storages[storage][cs.fields.begin] - 1:  # type: ignore
                    sts.logger.info("Reached end of storage by soft stop")
                    break

                i += 1

                if step.current_step() == total_steps - 1:
                    sts.logger.info("Reached end of storage by hard stop")
                    break

                while mpi_comm.iprobe(source=proceeder_rank, tag=MPI_TAGS.SERVICE):
                    sync_value = mpi_comm.recv(source=proceeder_rank, tag=MPI_TAGS.SERVICE)

                while worker_counter - sync_value > 50:
                    time.sleep(0.5)

    sts.logger.info("Reached end")
    mpi_comm.send(obj=1, dest=proceeder_rank, tag=MPI_TAGS.SERVICE)
    mpi_comm.send(obj=STATE.EXITED, dest=0, tag=MPI_TAGS.STATE)
    sts.logger.info("Exiting...")
    return 0
