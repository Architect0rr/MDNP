#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 06-09-2023 20:16:58

import json
from pathlib import Path
from typing import List, Dict, Any

import adios2
import numpy as np
from numpy import typing as npt

from .. import constants as cs
from ..core.distribution import distribution


def proceed(cwd: Path, storages: List, params: Dict, Natoms: int):
    worker_counter = 0
    max_cluster_size = 0
    sizes: npt.NDArray[np.uint32] = np.arange(1, Natoms + 1, dtype=np.uint32)
    for storage in storages:
        storage_fp = (cwd / params[cs.cf.dump_folder] / storage).as_posix()
        with adios2.open(storage_fp, 'r') as reader:  # type: ignore
            i = 0
        for step in reader:
            if i < storages[storage][cs.cf.begin]:  # type: ignore
                i += 1
                continue
            arr = step.read(cs.cf.lammps_dist)
            clusters = arr[:, 1].astype(dtype=np.uint32)
            dist = distribution(clusters, Natoms)

            # stepnd = worker_counter + ino

            # adout.write(cs.cf.mat_step, np.array(stepnd))  # type: ignore
            # adout.write(cs.cf.mat_dist, dist, dist.shape, np.full(len(dist.shape), 0), dist.shape, end_step=True)  # type: ignore

            max_cluster_size = np.argmax(sizes[dist != 0]) + 1

            worker_counter += 1

            if i == storages[storage][cs.cf.end] + storages[storage][cs.cf.begin] - 1:  # type: ignore
                print(f"Reached end of distribution, {storage, i, worker_counter}")
                break

            i += 1


def main():
    cwd = Path.cwd()
    data_file = (cwd / cs.files.data)
    with open(data_file, 'r') as fp:
        son: Dict[str, Any] = json.load(fp)
    _storages: List[str] = son[cs.cf.storages]
    pass


if __name__ == "__main__":
    import sys
    sys.exit(main())
