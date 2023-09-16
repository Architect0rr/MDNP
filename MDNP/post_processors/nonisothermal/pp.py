#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 16-09-2023 02:07:41

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Union

from MDDPN import constants as mcs
from . import constants as cs


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
            dump_file: Path = cwd / mcs.folders.dumps / rlabels[label][str(i)][mcs.sf.dump_file]
            if not dump_file.exists():
                fl = False
                logger.warning(f"Dump file {dump_file.as_posix()} not exists")
    return fl


def setup_logger(cwd: Path, logger: logging.Logger):
    folder = cwd / mcs.folders.log
    folder.mkdir(exist_ok=True, parents=True)
    folder = folder / "post"
    folder.mkdir(exist_ok=True, parents=True)
    logfile = folder / "post.log"

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


def end(cwd: Path, state: Dict[str, Any], args: argparse.Namespace, logger: logging.Logger) -> Union[Tuple[str, str], Tuple[None, None]]:
    logger = setup_logger(cwd, logger)

    if not (state_runs_check(state, logger.getChild('runs_check')) and state_validate(cwd, state, logger.getChild('validate'))):
        logger.error("Stopped, not valid state")
        return (None, None)

    df = []
    rlabels = state[mcs.sf.run_labels]

    for label in rlabels:
        for i in range(int(rlabels[label][mcs.sf.runs])):
            df.append(rlabels[label][str(i)][mcs.sf.dump_file])

    df = [f"{mcs.folders.dumps}/{el}" for el in df]

    if (stf := (cwd / cs.files.data)).exists():
        with open(stf, 'r') as fp:
            son = json.load(fp)
        son[cs.fields.storages] = df
        son[cs.fields.time_step] = state[mcs.sf.time_step]
        son[cs.fields.every] = state[mcs.sf.restart_every]
        son[cs.fields.data_processing_folder] = cs.folders.data_processing

    else:
        stf.touch()
        son = {
            cs.fields.storages: df,
            cs.fields.time_step: state[mcs.sf.time_step],
            cs.fields.every: state[mcs.sf.restart_every],
            cs.fields.data_processing_folder: cs.folders.data_processing}

    with open(stf, 'w') as fp:
        json.dump(son, fp)

    return "MDsimp", ""


if __name__ == "__main__":
    pass
