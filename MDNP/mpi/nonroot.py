#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 10-09-2023 08:14:40

from .utils_mpi import MC, MPI_TAGS
from .utils import Role
from .sense import workers


def goto(sts: MC):
    sts.mpi_comm.Barrier()
    mpi_comm = sts.mpi_comm
    mrole: Role = mpi_comm.recv(source=0, tag=MPI_TAGS.DISTRIBUTION)
    if mrole == Role.reader:
        mpi_comm.send(obj=str(mrole), dest=0, tag=MPI_TAGS.ONLINE)
        return workers.reader.reader(sts)
    elif mrole == Role.proceeder:
        mpi_comm.send(obj=str(mrole), dest=0, tag=MPI_TAGS.ONLINE)
        return workers.proceeder.proceed(sts)
    elif mrole == Role.treater:
        mpi_comm.send(obj=str(mrole), dest=0, tag=MPI_TAGS.ONLINE)
        return workers.treater.treat_mpi(sts)
    elif mrole == Role.killed:
        mpi_comm.send(obj=str(mrole), dest=0, tag=MPI_TAGS.ONLINE)
        mpi_comm.Barrier()
        return 0
    elif mrole == Role.one_thread:
        mpi_comm.send(obj=str(mrole), dest=0, tag=MPI_TAGS.ONLINE)
        return workers.one_threaded.thread(sts)
    elif mrole == Role.csvWriter:
        mpi_comm.send(obj=str(mrole), dest=0, tag=MPI_TAGS.ONLINE)
        return workers.writers.csvWriter(sts)
    elif mrole == Role.adios_writer:
        mpi_comm.send(obj=str(mrole), dest=0, tag=MPI_TAGS.ONLINE)
        return workers.writers.adios_writer(sts)
    elif mrole == Role.matr:
        mpi_comm.send(obj=str(mrole), dest=0, tag=MPI_TAGS.ONLINE)
        return workers.matrice.thread(sts)
    else:
        raise RuntimeError(f"Cannot find role {mrole}. Fatal error")
