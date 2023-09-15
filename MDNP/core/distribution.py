#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 16-09-2023 01:37:01


import freud  # type: ignore
import numpy as np
from numpy import typing as npt


def scale2box(points: npt.NDArray[np.float32], box: freud.box.Box) -> npt.NDArray[np.float32]:
    points[:, 0] = points[:, 0] * box.Lx
    points[:, 1] = points[:, 1] * box.Ly
    points[:, 2] = points[:, 2] * box.Lz
    return points


def clusters(points: npt.NDArray[np.float32], box: freud.box.Box, r_max: float) -> npt.NDArray[np.uint32]:
    points = box.wrap(points)
    system = freud.AABBQuery(box, points)
    cl = freud.cluster.Cluster()  # type: ignore
    cl.compute(system, neighbors={'mode': 'ball', "r_min": 0, 'exclude_ii': True, "r_max": r_max})
    cl_props = freud.cluster.ClusterProperties()  # type: ignore
    cl_props.compute(system, cl.cluster_idx)
    return cl_props.sizes


def distribution(data: npt.NDArray[np.uint32], N: int) -> npt.NDArray[np.uint32]:
    unique, counts = np.unique(data, return_counts=True)
    mat = np.zeros(N + 1, dtype=np.uint32)
    mat[unique] = counts
    return mat[1:]


def get_dist(data: npt.NDArray[np.float32], N: int, box: freud.box.Box) -> npt.NDArray[np.uint32]:
    return distribution(clusters(scale2box(data, box), box, 1.5), N)


if __name__ == "__main__":
    pass
