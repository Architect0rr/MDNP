#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 17-11-2023 02:15:10


import numpy as np


def nl(T: float) -> float:
    return 0.9367410354674542 * np.exp(-0.46391125909214476 * T**(2.791206046910478))


def sigma(T: float) -> float:
    return -1.8111682291065432 * T + 1.8524737887189553


def nvs(T: float) -> float:
    q = -1.9395800010433437*T + 6.089581679273376
    return 5.85002406256173*np.exp(-q/T)


def n1s(T: float) -> float:
    q = -1.10143417*T + 4.91381273
    return 2.05457139*np.exp(-q/T)


def nvs_reverse(nv: float) -> float:
    return -6.08958/np.log(0.02457499600439634*nv)


if __name__ == "__main__":
    pass
