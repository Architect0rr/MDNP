#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 17-09-2023 12:47:35


import logging
import os
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'


from mpi4py import MPI

from .root import main
from .nonroot import goto
from . import utils_mpi as UM
from .utils_mpi import MPISanityError, MC


def root(sts: MC):
    ret = UM.root_sanity(sts.mpi_comm)
    if ret != 0:
        sts.logger.critical("MPI root sanity doesn't passed")
        raise MPISanityError("MPI root sanity doesn't passed")
    else:
        sts.logger.info("Passed mpi root sanity check")
        return main(sts)


def nonroot(sts: MC):
    ret = UM.nonroot_sanity(sts.mpi_comm)
    if ret != 0:
        sts.logger.critical("MPI nonroot sanity doesn't passed")
        raise MPISanityError("MPI nonroot sanity doesn't passed")
    else:
        sts.logger.info("Passed mpi nonroot sanity check")
        return goto(sts)


def mpi_wrap():
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    cwd = Path.cwd()

    logger = UM.setup_logger(cwd, str(mpi_rank), logging.DEBUG)

    ret = UM.base_sanity(mpi_size, mpi_rank, 6)
    if ret != 0:
        logger.critical("MPI base sanity doesn't passed")
        raise MPISanityError("MPI base sanity doesn't passed")

    sts = MC(cwd, mpi_comm, mpi_rank, mpi_size, logger)

    if mpi_rank == 0:
        sts.logger = sts.logger.getChild('root')
        return root(sts)
    else:
        sts.logger = sts.logger.getChild('nonroot')
        return nonroot(sts)


if __name__ == "__main__":
    import sys
    sys.exit(mpi_wrap())
