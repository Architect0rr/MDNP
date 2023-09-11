#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 11-09-2023 20:56:30

import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore

from .. import constants as cs
from MDDPN import constants as mcs


def state_runs_check(state: dict) -> bool:
    fl = True
    rlabels = state[mcs.sf.run_labels]
    for label in rlabels:
        rc = 0
        while str(rc) in rlabels[label]:
            rc += 1
        prc = rlabels[label][mcs.sf.runs]
        if prc != rc:
            fl = False
            warnings.warn(f"Label {label} runs: present={prc}, real={rc}")
    return fl


def state_validate(cwd: Path, state: dict) -> bool:
    fl = True
    rlabels = state[mcs.sf.run_labels]
    for label in rlabels:
        for i in range(int(rlabels[label][mcs.sf.runs])):
            dump_file: Path = cwd / mcs.folders.dumps / rlabels[label][str(i)][mcs.sf.dump_file]
            if not dump_file.exists():
                fl = False
                warnings.warn(f"Dump file {dump_file.as_posix()} not exists")
    return fl


def calc_xi(xilog: Path, temps: Path) -> Tuple[float, int, int]:
    xist = pd.read_csv(xilog, header=None).to_numpy(dtype=np.int32).flatten()
    tf = pd.read_csv(temps, header=None)

    temp_time = tf[0].to_numpy(dtype=np.int32)
    temp_temp = tf[1].to_numpy(dtype=np.float32)

    temp1 = temp_temp[np.abs(temp_time - xist[0]) < 2][0]
    temp2 = temp_temp[np.abs(temp_time - xist[1]) < 2][0]
    # print(f"Temp1:{temp1},temp2:{temp2}")
    # print(f"Time1:{xist[0]},time2:{xist[1]}")

    return (float(np.abs((temp1 - temp2) / (xist[0] - xist[1]))), int(xist[0]), int(xist[1]))


def end(cwd: Path, state: Dict, args: argparse.Namespace, logger: logging.Logger) -> Tuple[str, Union[str, None]]:
    origin = cwd / cs.files.temperature
    origin.rename(cs.files.temperature_backup)
    origin = cwd / cs.files.temperature_backup
    target = cwd / cs.files.temperature
    target.touch()
    with origin.open('r') as fin, target.open('w') as fout:
        for line in fin:
            if line[0] == '#':
                continue
            fout.write(line)

    if not (state_runs_check(state) and state_validate(cwd, state)):
        print("Stopped, not valid state")

    xi, step_before, step_after = calc_xi(cwd / cs.files.xi_log, target)
    xi = xi / state[mcs.sf.time_step]
    print(f"XI: {xi}")

    df = []
    rlabels = state[mcs.sf.run_labels]

    for label in rlabels:
        for i in range(int(rlabels[label][mcs.sf.runs])):
            df.append(rlabels[label][str(i)][mcs.sf.dump_file])

    if (stf := (cwd / cs.files.data)).exists():
        with open(stf, 'r') as fp:
            son = json.load(fp)
        son[cs.fields.xi] = xi
        son[cs.fields.step_before] = step_before
        son[cs.fields.step_after] = step_after
        son[cs.fields.storages] = df
        son[cs.fields.time_step] = state[mcs.sf.time_step]
        son[cs.fields.every] = state[mcs.sf.restart_every]
        # son[cs.fields.dump_folder] = cs.folders.dumps
        son[cs.fields.data_processing_folder] = cs.folders.data_processing

    else:
        stf.touch()
        son = {
            cs.fields.step_before: step_before,
            cs.fields.step_after: step_after,
            cs.fields.xi: xi,
            cs.fields.storages: df,
            cs.fields.time_step: state[mcs.sf.time_step],
            cs.fields.every: state[mcs.sf.restart_every],
            # cs.fields.dump_folder: cs.folders.dumps,
            cs.fields.data_processing_folder: cs.folders.data_processing}

    with open(stf, 'w') as fp:
        json.dump(son, fp)

    executable = Path("/scratch/perevoshchikyy/MD/MDDPN/launcher.py")

    return executable.as_posix(), None


if __name__ == "__main__":
    pass
