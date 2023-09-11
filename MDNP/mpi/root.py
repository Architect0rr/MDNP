#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 10-09-2023 16:46:22

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any

import adios2
import numpy as np
from numpy import typing as npt

from .utils_mpi import MC
from .. import constants as cs
from .sense.root.group import group_run
from .sense.root.one_threaded import one_threaded
from .sense.root.new import new


def bearbeit(folder: Path, storages: Dict[str, int]) -> Tuple[int, npt.NDArray[np.float32]]:
    storage = list(storages.keys())[0]
    adin = adios2.open((folder / storage).as_posix(), 'r')  # type: ignore

    N = int(adin.read(cs.lcf.natoms))
    Lx = float(adin.read(cs.lcf.boxxhi))
    Ly = float(adin.read(cs.lcf.boxyhi))
    Lz = float(adin.read(cs.lcf.boxzhi))

    adin.close()

    bdims = np.array([Lx, Ly, Lz])
    return (N, bdims)


def storage_rsolve(cwd: Path, _storages: List[str]) -> Dict[str, int]:
    storages = {}
    for storage in _storages:
        file = cwd / storage
        if not file.exists():
            raise FileNotFoundError(f"Storage {file.as_posix()} cannot be found")
        with adios2.open(file.as_posix(), 'r') as reader_c:  # type: ignore
            storages[storage] = reader_c.steps()
    return storages


def main(sts: MC):
    sts.logger.info("Started")

    parser = argparse.ArgumentParser(description='Generate cluster distribution matrix from ADIOS2 LAMMPS data.')
    parser.add_argument('--debug', action='store_true', help='Debug, prints only parsed arguments')
    parser.add_argument('--mode', action='store', type=int, default=3, help='Mode to run')
    args = parser.parse_args()

    sts.logger.info(f"Envolved args: {args}")

    if args.debug:
        sts.logger.setLevel(logging.DEBUG)
    else:
        sts.logger.setLevel(logging.INFO)

    sts.logger.debug("Reading info file")
    data_file = sts.cwd / cs.files.data
    with open(data_file, 'r') as fp:
        son: Dict[str, Any] = json.load(fp)
    _storages: List[str] = son[cs.fields.storages]
    sts.logger.debug("Resolving storages")
    storages = storage_rsolve(sts.cwd, _storages)

    sts.logger.debug("Getting N of atoms and dimensions")
    N_atoms, bdims = bearbeit(sts.cwd, storages)
    son[cs.fields.N_atoms] = N_atoms
    son[cs.fields.volume] = np.prod(bdims)
    son[cs.fields.dimensions] = list(bdims)
    son[cs.fields.storages] = storages
    sts.logger.debug("Updating info file")
    with open(data_file, 'w') as fp:
        json.dump(son, fp)

    if args.mode == 1:
        sts.logger.info("Running group run")
        sts.logger = sts.logger.getChild('group')
        return group_run(sts, son, 3)
    elif args.mode == 2:
        sts.logger = sts.logger.getChild('one')
        sts.logger.info("Running one threaded run")
        return one_threaded(sts, son, 1)
    elif args.mode == 3:
        sts.logger = sts.logger.getChild('new')
        sts.logger.info("Running new run")
        return new(sts, son, 1)
