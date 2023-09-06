#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 06-09-2023 20:50:08

import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict

from ..control.utils import states
from .. import constants as cs
from ..control.execution import perform_processing_run


def state_runs_check(state: dict) -> bool:
    fl = True
    rlabels = state[cs.sf.run_labels]
    for label in rlabels:
        rc = 0
        while str(rc) in rlabels[label]:
            rc += 1
        prc = rlabels[label][cs.sf.runs]
        if prc != rc:
            fl = False
            warnings.warn(f"Label {label} runs: present={prc}, real={rc}")
    return fl


def state_validate(cwd: Path, state: dict) -> bool:
    fl = True
    rlabels = state[cs.sf.run_labels]
    for label in rlabels:
        for i in range(int(rlabels[label][cs.sf.runs])):
            dump_file: Path = cwd / cs.folders.dumps / rlabels[label][str(i)][cs.sf.dump_file]
            if not dump_file.exists():
                fl = False
                warnings.warn(f"Dump file {dump_file.as_posix()} not exists")
    return fl


def setup_logger(cwd: Path, logger: logging.Logger):
    folder = cwd / cs.folders.log
    folder.mkdir(exist_ok=True, parents=True)
    folder = folder / cs.folders.post_process_log
    folder.mkdir(exist_ok=True, parents=True)
    logfile = folder / "end.log"

    handler = logging.FileHandler(logfile)
    handler.setFormatter(cs.sp.formatter)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.WARNING)

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.ERROR)

    logger.addHandler(handler)
    logger.addHandler(stdout)
    logger.addHandler(stderr)

    return logger


def end(cwd: Path, state: Dict, args: argparse.Namespace, logger: logging.Logger) -> Dict:
    logger = setup_logger(cwd, logger)

    if not (state_runs_check(state) and state_validate(cwd, state)):
        logger.error("Stopped, not valid state")
        return state

    df = []
    rlabels = state[cs.sf.run_labels]

    for label in rlabels:
        for i in range(int(rlabels[label][cs.sf.runs])):
            df.append(rlabels[label][str(i)][cs.sf.dump_file])

    if (stf := (cwd / cs.files.data)).exists():
        with open(stf, 'r') as fp:
            son = json.load(fp)
        son[cs.cf.storages] = df
        son[cs.sf.time_step] = state[cs.sf.time_step]
        son[cs.sf.restart_every] = state[cs.sf.restart_every]
        son[cs.cf.dump_folder] = cs.folders.dumps
        son[cs.cf.data_processing_folder] = cs.folders.data_processing

    else:
        stf.touch()
        son = {
            cs.cf.storages: df,
            cs.sf.time_step: state[cs.sf.time_step],
            cs.sf.restart_every: state[cs.sf.restart_every],
            cs.cf.dump_folder: cs.folders.dumps,
            cs.cf.data_processing_folder: cs.folders.data_processing}

    with open(stf, 'w') as fp:
        json.dump(son, fp)

    job_id = perform_processing_run(cwd, state, args.params, args.part, args.nodes)

    state[cs.sf.post_process_id] = job_id

    state[cs.sf.state] = states.data_obtained
    return state


if __name__ == "__main__":
    pass
