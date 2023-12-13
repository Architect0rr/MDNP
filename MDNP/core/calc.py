#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 17-11-2023 02:28:58


import numpy as np
from typing import Union, List
from numpy import typing as npt

from . import props
from ..utils import fp


def mean_size(sizes: npt.NDArray[np.uint32], dist: npt.NDArray[np.uint32]) -> fp:
    return np.sum(sizes * dist) / np.sum(dist)


def condensation_degree(dist: npt.NDArray[np.uint32], sizes: npt.NDArray[np.uint32], N: int, km: int) -> fp:
    return 1 - np.sum(sizes[:km] * dist[:km]) / N


def maxsize(sizes: npt.NDArray[np.uint32], dist: npt.NDArray[np.uint32]) -> int:
    return sizes[np.nonzero(dist)][-1]  # type: ignore


def nvv(sizes: npt.NDArray[np.uint32], dist: npt.NDArray[np.uint32], volume: float, kmin: int) -> fp:
    # type: ignore
    return np.sum(dist[sizes <= kmin] * sizes[sizes <= kmin]) / volume


def nd(sizes: npt.NDArray[np.uint32], dist: npt.NDArray[np.uint32], volume: float, kmin: int) -> fp:
    return np.sum(dist[sizes >= kmin]) / volume


def nvs(sizes: npt.NDArray[np.uint32], dist: npt.NDArray[np.uint32], volume: float, kmin: int, T: float) -> Union[float, None]:
    ms = sizes[-1]
    n1 = dist[0] / volume
    dzd = dist[kmin - 1:ms]
    kks = np.arange(kmin, ms + 1, dtype=np.uint32)**(1 / 3)
    num = n1 * np.sum(kks**2 * dzd) / volume
    if num == 0:
        return None
    rl = (3 / (4 * np.pi * props.nl(T)))**(1 / 3)
    cplx = 2 * props.sigma(T) / (props.nl(T) * T * rl)
    denum = np.sum(kks**2 * dzd * np.exp(cplx / kks)) / volume

    return num / denum


def Srh_props(nvv: fp, T: float) -> fp:
    return nvv/props.nvs(T)


def get_spec() -> List[str]:
    return ['step', 'time', 'x', 'nv', 'T', 'nvs', 'Srh', 'Srh_p', 'nd', 'S1']


def get_row(step: int, sizes: npt.NDArray[np.uint32], dist: npt.NDArray[np.uint32], T: float, N_atoms: int, volume: float, dt: float, dis: int, kmin: int) -> npt.NDArray[np.float32]:
    # km: int = 10
    # eps = 0.9

    # ld = np.array([np.sum(sizes[:i]*dist[:i]) / N_atoms for i in range(1, len(dist))], dtype=np.float32)
    # km = np.argmin(np.abs(ld - eps))

    # tow = np.zeros(8, dtype=np.float32)
    nv: fp = nvv(sizes, dist, volume, kmin)
    nvss = nvs(sizes, dist, volume, kmin, T)
    if nvss is None:
        Srh = 0
    else:
        Srh = nv / nvss
    try:
        S = dist[0] / props.n1s(T) / volume
    except Exception:
        S = 0
    tow: List[fp | None] = [
        step,
        round(step * dt * dis),
        condensation_degree(dist, sizes, N_atoms, km=kmin),
        nv,
        T,
        nvss,
        Srh,
        Srh_props(nv, T),
        nd(sizes, dist, volume, kmin),
        S
    ]
    # tow[1] = mean_size(sizes, dist)
    # tow[2] = maxsize(sizes, dist)
    # tow[5] = nd(sizes, dist, volume, g)
    # tow[6] = len(dist[dist > 1])
    # tow[9] = np.sum(dist[g-1:])
    return np.array(tow, dtype=np.float32)


if __name__ == "__main__":
    pass
