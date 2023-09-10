#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 10-09-2023 01:05:20


import os
import sys
import time
import secrets
import logging
from enum import Enum
from pathlib import Path
from typing import Literal, Union

os.environ['OPENBLAS_NUM_THREADS'] = '1'

from mpi4py import MPI

from .. import constants as cs
from .utils import MPIComm, GatherResponseType


class MC():  # MPI conf
    def __init__(self, cwd: Path, mpi_comm: MPIComm, mpi_rank: int, mpi_size: int, logger: logging.Logger) -> None:
        self.cwd: Path = cwd
        self.mpi_comm: MPIComm = mpi_comm
        self.mpi_rank: int = mpi_rank
        self.mpi_size: int = mpi_size
        self.logger: logging.Logger = logger
        pass


class MPI_TAGS(int, Enum):
    SANITY = 0
    DISTRIBUTION = 1
    COMMAND = 2
    SERV_DATA = 3
    TO_ACCEPT = 4
    WRITE = 5
    DATA = 6
    SERVICE = 7
    STATE = 8
    ONLINE = 9
    SERV_DATA_1 = 10
    SERV_DATA_2 = 11
    SERV_DATA_3 = 12


def blockPrint() -> None:
    sys.stdout = open(os.devnull, 'w')


def enablePrint() -> None:
    sys.stdout = sys.__stdout__


class MPISanityError(RuntimeError):
    pass


def setup_logger(cwd: Path, name: str, level: int = logging.INFO) -> logging.Logger:
    folder = cwd / cs.folders.log
    folder.mkdir(exist_ok=True, parents=True)
    folder = folder / cs.folders.mpi_log
    folder.mkdir(exist_ok=True, parents=True)

    logfile = folder / f"{name}.log"

    handler = logging.FileHandler(logfile)
    handler.setFormatter(cs.obj.formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def base_sanity(mpi_size: int, mpi_rank: int, min: int) -> Literal[0]:
    if mpi_size == 1:
        print('You are running an MPI program with only one slot/task!')
        print('Are you using `mpirun` (or `srun` when in SLURM)?')
        print('If you are, then please use an `-n` of at least 2!')
        print('(Or, when in SLURM, use an `--ntasks` of at least 2.)')
        print('If you did all that, then your MPI setup may be bad.')
        raise MPISanityError("Only one execution thread was started")

    if mpi_size < min:
        print(
            f"This program requires at least {min} mpi tasks, but world size is only {mpi_size}")
        raise MPISanityError(f"Number of started threads is not enought to properly run this app. You must run at least {min} threads")

    if mpi_size >= 1000 and mpi_rank == 0:
        print('WARNING:  Your world size {} is over 999!'.format(mpi_size))
        print("The output formatting will be a little weird, but that's it.")

    return 0


def root_sanity(mpi_comm: MPIComm) -> Literal[1, 0]:
    random_number = secrets.randbelow(round(time.time()))
    mpi_comm.bcast(random_number)
    print('Controller @ MPI Rank   0:  Input {}'.format(random_number))

    response_array: GatherResponseType = mpi_comm.gather(None)  # type: ignore

    mpi_size: int = mpi_comm.Get_size()
    if len(response_array) != mpi_size:
        print(f"ERROR!  The MPI world has {mpi_size} members, but we only gathered {len(response_array)}!")
        return 1

    for i in range(1, mpi_size):
        if len(response_array[i]) != 2:
            print(f"WARNING!  MPI rank {i} sent a mis-sized ({len(response_array[i])}) tuple!")
            continue
        if type(response_array[i][0]) is not str:
            print(f"WARNING!  MPI rank {i} sent a tuple with a {str(type(response_array[i][0]))} instead of a str!")
            continue
        if type(response_array[i][1]) is not int:
            print(f"WARNING!  MPI rank {i} sent a tuple with a {str(type(response_array[i][1]))} instead of an int!")
            continue

        if random_number + i == response_array[i][1]:
            result = 'OK'
        else:
            result = 'BAD'

        print(f"Worker at MPI Rank {i}: Output {response_array[i][1]} is {result} (from {response_array[i][0]})")

        mpi_comm.send(obj=0, dest=i, tag=MPI_TAGS.SANITY)

    return 0


def nonroot_sanity(mpi_comm: MPIComm) -> Literal[1, 0]:
    mpi_rank: int = mpi_comm.Get_rank()

    random_number: int = mpi_comm.bcast(None)

    # Sanity check: Did we actually get an int?
    if type(random_number) is not int:
        print(
            f"ERROR in MPI rank {mpi_rank}: Received a non-integer '{random_number}' from the broadcast!")
        return 1

    # Our response is the random number + our rank
    response_number: int = random_number + mpi_rank
    response: tuple[str, int] = (
        MPI.Get_processor_name(),
        response_number,
    )
    mpi_comm.gather(response)

    def get_message(mpi_comm: MPIComm) -> Union[int, None]:
        message: int = mpi_comm.recv(source=0, tag=MPI_TAGS.SANITY)
        if type(message) is not int:
            print(
                f"ERROR in MPI rank {mpi_rank}: Received a non-integer message!")
            return None
        else:
            return message

    message: Union[int, None] = get_message(mpi_comm)
    while (message is not None) and (message != 0):
        mpi_comm.send(obj=int(message / 2), dest=0, tag=MPI_TAGS.SANITY)
        message = get_message(mpi_comm)

    # Did we get an error?
    if message is None:
        return 1
    return 0


if __name__ == "__main__":
    pass
