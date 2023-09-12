#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 12-09-2023 12:28:02

import csv
import platform
from enum import Enum
from pathlib import Path
from typing import Generator, Any
from contextlib import contextmanager

py_ver_maj, py_ver_min, py_ver_patch = platform.python_version().split(".")
if int(py_ver_maj) == 3:
    if int(py_ver_min) >= 11:
        from typing import Self  # type: ignore
    else:
        from typing_extensions import Self
else:
    raise Exception("Python version must be at least 3.6")

import adios2


class wmode(Enum):
    csv = 1
    adios = 2
    both = 3


class Writer:
    def __init__(self, ad_storage: Path, csv_storage: Path, mode: wmode) -> None:
        self.adios_fp = ad_storage
        self.csv_fp = csv_storage
        self.wmode = mode

    def __open_adios(self) -> None:
        self.adios_fh = adios2.open(self.adios_fp.as_posix(), 'w')  # type: ignore

    def __close_adios(self) -> None:
        self.adios_fh.close()

    def __close_csv(self) -> None:
        self.csv_fh.close()

    def __open_csv(self) -> None:
        self.csv_fh = self.csv_fp.open('w')
        self.csv_writer = csv.writer(self.csv_fh, delimiter=',')

    def __write_adios(self, data) -> None:
        pass

    def __write_csv(self, data) -> None:
        pass

    def __write_both(self, data):
        self.__write_adios(data)
        self.__write_csv(data)

    def __write(self, data) -> None:
        pass

    def write(self, data) -> None:
        return self.__write(data)

    @contextmanager
    def wopen(self) -> Generator[Self, Self, None]:
        if self.wmode == wmode.adios:
            self.__open_adios()
            self.__write = self.__write_adios
        elif self.wmode == wmode.csv:
            self.__open_csv()
            self.__write = self.__write_csv
        elif self.wmode == wmode.both:
            self.__open_adios()
            self.__open_csv()
            self.__write = self.__write_both

        try:
            yield self
        finally:

            if self.wmode == wmode.adios:
                self.__close_adios()
            elif self.wmode == wmode.csv:
                self.__close_csv()
            elif self.wmode == wmode.both:
                self.__close_adios()
                self.__close_csv()
