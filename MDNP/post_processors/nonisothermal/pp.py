#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 13-12-2023 19:37:12

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Union

import adios2
import numpy as np
from numpy import typing as npt

from MDDPN import constants as mcs
from . import constants as cs


def bearbeit(storage: Path) -> Tuple[int, npt.NDArray[np.float32]]:
    adin = adios2.open(storage.as_posix(), 'r')  # type: ignore

    N = int(adin.read(cs.lcf.natoms))
    Lx = float(adin.read(cs.lcf.boxxhi))
    Ly = float(adin.read(cs.lcf.boxyhi))
    Lz = float(adin.read(cs.lcf.boxzhi))

    adin.close()

    bdims = np.array([Lx, Ly, Lz])
    return (N, bdims)


def state_runs_check(state: dict, logger: logging.Logger) -> bool:
    fl = True
    rlabels = state[mcs.sf.run_labels]
    for label in rlabels:
        rc = 0
        while str(rc) in rlabels[label]:
            rc += 1
        prc = rlabels[label][mcs.sf.runs]
        if prc != rc:
            fl = False
            logger.warning(f"Label {label} runs: present={prc}, real={rc}")
    return fl


def state_validate(cwd: Path, state: dict, logger: logging.Logger) -> bool:
    fl = True
    rlabels = state[mcs.sf.run_labels]
    for label in rlabels:
        for i in range(int(rlabels[label][mcs.sf.runs])):
            logger.debug(f"Checking {label}:{i}:{mcs.sf.dump_file}")
            try:
                dump_file: Path = cwd / mcs.folders.dumps / rlabels[label][str(i)][mcs.sf.dump_file]
            except KeyError:
                logging.exception(json.dumps(rlabels, indent=4))
                raise
            if not dump_file.exists():
                fl = False
                logger.warning(f"Dump file {dump_file.as_posix()} not exists")
    return fl


def setup_logger(cwd: Path, logger: logging.Logger):
    folder = cwd / mcs.folders.log
    folder.mkdir(exist_ok=True, parents=True)
    folder = folder / cs.folders.post_process_log
    folder.mkdir(exist_ok=True, parents=True)
    logfile = folder / "run.log"

    handler = logging.FileHandler(logfile)
    handler.setFormatter(cs.obj.formatter)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.WARNING)

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.ERROR)

    logger.addHandler(handler)
    logger.addHandler(stdout)
    logger.addHandler(stderr)

    return logger


def end(cwd: Path, state: Dict[str, Any], args: argparse.Namespace, logger: logging.Logger, anyway: bool = False) -> Union[Tuple[str, str], Tuple[None, None]]:
    logger = setup_logger(cwd, logger)

    if not anyway:
        if not (state_runs_check(state, logger.getChild('runs_check')) and state_validate(cwd, state, logger.getChild('validate'))):
            logger.error("Stopped, not valid state")
            return (None, None)

    df = []
    rlabels = state[mcs.sf.run_labels]

    logger.info("Getting storages from state")
    for label in rlabels:
        for i in range(int(rlabels[label][mcs.sf.runs])):
            df.append(rlabels[label][str(i)][mcs.sf.dump_file])

    gf = [f"{mcs.folders.dumps}/{el}" for el in df]

    df = []
    for el in gf:
        if (cwd / el).exists():
            df.append(el)

    logger.info("Getting info from first storage")
    sto_check: Path = cwd / df[0]
    Natoms, dims = bearbeit(sto_check)

    son = {
        cs.fields.storages: df,
        cs.fields.time_step: state[mcs.sf.time_step],
        cs.fields.every: state[mcs.sf.restart_every],
        cs.fields.data_processing_folder: cs.folders.data_processing,
        cs.fields.N_atoms: Natoms,
        cs.fields.dimensions: list(dims),
        cs.fields.volume: float(np.prod(dims))}

    if (stf := (cwd / cs.files.data)).exists():
        logger.info(f"Datafile {stf.as_posix()} already existed, deleting")
        stf.unlink()
        # with stf.open('r') as fp:
        #     son = json.load(fp)
        # son[cs.fields.storages] = df
        # son[cs.fields.time_step] = state[mcs.sf.time_step]
        # son[cs.fields.every] = state[mcs.sf.restart_every]
        # son[cs.fields.data_processing_folder] = cs.folders.data_processing
        # son[cs.fields.N_atoms] = Natoms
        # son[cs.fields.dimensions] = list(dims)
        # son[cs.fields.volume] = float(np.prod(dims))

    logger.info("Writing parameters to datafile")
    with open(stf, 'w') as fp:
        json.dump(son, fp)

    logger.info("Returning executable and args")
    return "MDpost_run", " --mode=4"


if __name__ == "__main__":
    pass
