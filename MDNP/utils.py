#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 09-09-2023 01:00:38

from typing import Union, Iterable
import numpy as np
from numpy import typing as npt


fp = Union[float, np.floating]


def is_iter(arr: Union[Iterable[float], float]) -> bool:
    try:
        iter(arr)  # type: ignore
        return True
    except Exception:
        return False
