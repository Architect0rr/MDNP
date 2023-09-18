#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 18-09-2023 21:51:13

from typing import Dict, Any

import adios2
import numpy as np
from numpy import typing as npt

from .utils_mpi import MC


class adser:
    def __init__(self, sts: MC, mpi: bool = False) -> None:
        # self.sts = sts
        self.mpi = mpi
        self.logger = sts.logger.getChild('adios_lib')
        self.logger.debug("Creating ADIOS instance")
        if self.mpi:
            self.rank = sts.mpi_rank
            self.size = sts.mpi_size
            self.adios = adios2.ADIOS(sts.mpi_comm)  # type: ignore
        else:
            self.rank = 0
            self.size = 1
            self.adios = adios2.ADIOS()  # type: ignore
        self.logger.debug("Declaring IO")
        self.bpIO = self.adios.DeclareIO("BPFile_N2N")
        self.logger.debug("Setting engine")
        self.bpIO.SetEngine('bp5')
        self.logger.debug("Adding transport")
        self.fileID = self.bpIO.AddTransport('File', {'Library': 'POSIX'})
        self.vars_arr: Dict[str, Any] = {}
        self.vars_one: Dict[str, Any] = {}
        self.opened = False
        self.step = 0

        # return adwriter, ad_sizes, ad_size_counts, ad_temps, ad_dist

    def open(self, name: str) -> None:
        if self.opened:
            self.logger.error("Attempt to open storage in second time")
            raise RuntimeError("Attempt to open storage in second time")
        self.logger.debug(f"Opening storage {name}")
        self.adwriter = self.bpIO.Open(name, adios2.Mode.Write)  # type: ignore

    def close(self):
        self.logger.debug("Closing storage")
        self.adwriter.Close()

    def declare_arr(self, name: str, Nx: int, dtype):
        self.logger.debug(f"Declaring 1D variable '{name}' with size {Nx} with type {dtype}")
        templ = np.zeros(Nx, dtype=dtype)
        var = self.bpIO.DefineVariable(
            name,                 # name
            templ,                # array
            [self.size * Nx],     # shape
            [self.rank * Nx],     # start
            [Nx],                 # count
            adios2.ConstantDims)  # type: ignore # constantDims
        self.vars_arr[name] = (Nx, var, dtype)

    def declare_2d_arr(self, name: str, Nx: int, Ny: int, dtype):
        self.logger.debug(f"Declaring 2D variable '{name}' with size {Nx}x{Ny} with type {dtype}")
        templ = np.zeros((Ny, Ny), dtype=dtype)
        var = self.bpIO.DefineVariable(
            name,                  # name
            templ,                 # array
            [self.size * Nx, Ny],  # shape
            [self.rank * Nx, 0],   # start
            [Nx, Ny],              # count
            adios2.ConstantDims)   # type: ignore # constantDims
        self.vars_arr[name] = ((Nx, Ny), var, dtype)

    def declare_one(self, name: str, dtype):
        self.logger.debug(f"Declaring 0D variable '{name}' with type {dtype}")
        self.vars_one[name] = (
            self.bpIO.DefineVariable(name, np.zeros(1, dtype=dtype)),
            dtype)

    def begin_step(self):
        self.logger.debug(f"Beginning step {self.step}")
        self.adwriter.BeginStep()

    def end_step(self):
        self.logger.debug(f"Ending step {self.step}")
        self.step += 1
        self.adwriter.EndStep()

    def wr_array(self, name: str, arr: npt.NDArray):
        self.logger.debug(f"Writing variable '{name}'")
        Nx, var, dtype = self.vars_arr[name]
        arr = np.resize(arr, Nx)
        arr = arr.astype(dtype=dtype)
        self.adwriter.Put(var, arr)

    def wr_2d_array(self, name: str, arr: npt.NDArray):
        self.logger.debug(f"Writing variable '{name}'")
        shape, var, dtype = self.vars_arr[name]
        arr = np.resize(arr, shape)
        arr = arr.astype(dtype=dtype)
        self.adwriter.Put(var, arr)

    def wr_one(self, name: str, val):
        self.logger.debug(f"Writing variable '{name}'")
        var, dtype = self.vars_one[name]
        self.adwriter.Put(var, np.array([val], dtype=dtype))
