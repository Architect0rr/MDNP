#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 09-09-2023 01:37:11


import sys

from MDNP.mpi import multmpi


sys.exit(multmpi.mpi_wrap())
