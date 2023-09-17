#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 17-09-2023 12:29:03

from .utils import Role
from .utils import STATE
from .sense import workers
from .utils_mpi import MC, MPI_TAGS


def w4sb(sts: MC):  # wait for second barrier
    sts.logger.info("Waiting for second barrier")
    sts.mpi_comm.Barrier()
    sts.logger.info("Second barrier released")


def goto(sts: MC):
    sts.logger.info("Waiting for distribution barrier")
    sts.mpi_comm.Barrier()
    sts.logger.info("Barrier released")
    mpi_comm = sts.mpi_comm
    sts.logger.info("Receiving role")
    mrole: Role = mpi_comm.recv(source=0, tag=MPI_TAGS.DISTRIBUTION)
    sts.logger.info(f"Role '{mrole}' received")

    # group
    if mrole == Role.csvWriter:
        mpi_comm.send(obj=mrole, dest=0, tag=MPI_TAGS.ONLINE)
        sts.logger = sts.logger.getChild('csvWriter')
        w4sb(sts)
        return workers.writers.csvWriter(sts)
    elif mrole == Role.adios_writer:
        mpi_comm.send(obj=mrole, dest=0, tag=MPI_TAGS.ONLINE)
        sts.logger = sts.logger.getChild('adios_writer')
        w4sb(sts)
        return workers.writers.adios_writer(sts)
    elif mrole == Role.reader:
        mpi_comm.send(obj=mrole, dest=0, tag=MPI_TAGS.ONLINE)
        sts.logger = sts.logger.getChild('reader')
        w4sb(sts)
        return workers.reader.reader(sts)
    elif mrole == Role.proceeder:
        mpi_comm.send(obj=mrole, dest=0, tag=MPI_TAGS.ONLINE)
        sts.logger = sts.logger.getChild('proceeder')
        w4sb(sts)
        return workers.proceeder.proceed(sts)
    elif mrole == Role.treater:
        mpi_comm.send(obj=mrole, dest=0, tag=MPI_TAGS.ONLINE)
        sts.logger = sts.logger.getChild('treater')
        w4sb(sts)
        return workers.treater.treat_mpi(sts)

    elif mrole == Role.killed:
        mpi_comm.send(obj=mrole, dest=0, tag=MPI_TAGS.ONLINE)
        sts.logger = sts.logger.getChild('killed')
        w4sb(sts)
        mpi_comm.send(obj=STATE.EXITED, dest=0, tag=MPI_TAGS.STATE)
        return 0
    # one threaded
    elif mrole == Role.one_thread:
        mpi_comm.send(obj=mrole, dest=0, tag=MPI_TAGS.ONLINE)
        sts.logger = sts.logger.getChild('one_thread')
        w4sb(sts)
        return workers.one_threaded.thread(sts)
    # new
    elif mrole == Role.matr:
        mpi_comm.send(obj=mrole, dest=0, tag=MPI_TAGS.ONLINE)
        sts.logger = sts.logger.getChild('matr')
        w4sb(sts)
        return workers.matrice.thread(sts)
    elif mrole == Role.simple:
        mpi_comm.send(obj=mrole, dest=0, tag=MPI_TAGS.ONLINE)
        sts.logger = sts.logger.getChild('simp')
        w4sb(sts)
        return workers.simp.simple(sts)
    else:
        raise RuntimeError(f"Cannot find role {mrole}. Fatal error")
