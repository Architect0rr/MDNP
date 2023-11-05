#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 04-11-2023 09:25:42

import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
from numpy import typing as npt

from ..core import calc, props
from .. import constants as cs


BUF_SIZE = 65536  # 64kb


def find_file(cwd: Path, file: Union[str, None], subf: str, defname: str) -> Path:
    if (cwd / subf).exists():
        if file is None:
            if (f := (cwd / subf / defname)).exists():
                return f
            raise FileNotFoundError(f"File {f.as_posix()} cannot be found")
        elif (f := (cwd / subf / file)).exists():
            return f
        raise FileNotFoundError(f"File {f.as_posix()} cannot be found")
    elif file is None:
        if (f := (cwd / defname)).exists():
            return f
        raise FileNotFoundError(f"File {f.as_posix()} cannot be found")
    elif (f := (cwd / file)).exists():
        return f
    raise FileNotFoundError(f"File {f.as_posix()} cannot be found")


def proceed(infile: Path, outfile: Path, conf: Dict, temp_mat: Tuple[npt.NDArray[np.uint64], npt.NDArray[np.float32]], cut: int, km: int):
    temptime, temperatures = temp_mat
    dis = conf[cs.fields.every]
    N_atoms = conf[cs.fields.N_atoms]
    time_step = conf[cs.fields.time_step]
    volume = conf[cs.fields.volume]
    sizes: npt.NDArray[np.uint32] = np.arange(1, cut + 1, 1, dtype=np.uint32)
    with pd.read_csv(infile, header=None, chunksize=BUF_SIZE) as reader, open(outfile, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(calc.get_spec())
        chunk: pd.DataFrame
        for chunk in reader:
            for index, row in chunk.iterrows():
                step = int(row[0])
                dist = row[1:].to_numpy(dtype=np.uint32)
                # print(f"Searching for {int(step * dis)}")
                temp: float = temperatures[temptime == int(step)][0]  # type: ignore
                tow = calc.get_row(step, sizes, dist, temp, N_atoms, volume, time_step, dis, km)
                writer.writerow(tow)

    return 0


def ncut(infile: Path) -> int:
    with infile.open("r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            return len(line) - 1
    return 0


def km(infile: Path, conf: Dict, cut: int, eps: float) -> int:
    fst = round(conf[cs.fields.step_before] / conf[cs.fields.every])
    N_atoms = conf[cs.fields.N_atoms]
    sizes = np.arange(1, cut + 1, 1)
    with pd.read_csv(infile, header=None, chunksize=BUF_SIZE) as reader:
        chunk: pd.DataFrame
        for chunk in reader:
            index = chunk[0].to_numpy(dtype=np.uint32)
            if len(index[index == fst]) == 1:
                ind = np.where(index == fst)
                dist = chunk.iloc[ind].to_numpy(dtype=np.uint32)[0][1:]  # type: ignore

                ld = np.array([np.sum(sizes[:i]*dist[:i]) / N_atoms for i in range(1, len(dist))], dtype=np.float32)
                km = np.argmin(np.abs(ld - eps))
                return int(km)
    raise KeyError(f"Step before {conf[cs.fields.step_before]}/{conf[cs.fields.every]}={fst} not found in matrix")


def get_spec_step(infile: Path, needed_step: int) -> npt.NDArray[np.uint64]:
    with pd.read_csv(infile, header=None, chunksize=BUF_SIZE) as reader:
        chunk: pd.DataFrame
        for chunk in reader:
            for index, row in chunk.iterrows():
                step = int(row[0])
                if step == needed_step:
                    dist = row[1:].to_numpy(dtype=np.uint32)
                    return dist
    raise Exception("Specified step not found")


def get_S1_dist(cwd: Path, son: Dict, data_file: Path, dis: int) -> npt.NDArray[np.uint64]:
    temperatures_mat = pd.read_csv(cwd / cs.files.temperature, header=None)
    temptime: npt.NDArray[np.uint64] = temperatures_mat[0].to_numpy(dtype=np.uint64)
    temperatures: npt.NDArray[np.float32] = temperatures_mat[1].to_numpy(dtype=np.float32)

    N_atoms: int = son[cs.fields.N_atoms]
    nvz: float = N_atoms/son[cs.fields.volume]
    Stemp: float = props.nvs_reverse(nvz)
    Sstep_var: float = temptime[np.argmin(np.abs(temperatures - Stemp))]
    Sstep: int = int(Sstep_var/dis) + 1

    return get_spec_step(data_file, Sstep)


def get_named_moment(cwd: Path, son: Dict, data_file: Path, val: str, dis: int) -> npt.NDArray[np.uint64]:
    if val == "S1":
        return get_S1_dist(cwd, son, data_file, dis)
    else:
        raise Exception(f"Unknown named moment: {val}")


def dist_getter(cwd: Path, args: argparse.Namespace, son: Dict, data_file: Path, dt: float, dis: int, sizes: npt.NDArray[np.uint64], cut: int):
    if args.method == 'name':
        dist: npt.NDArray[np.uint64] = get_named_moment(cwd, son, data_file, str(args.value), dis)
    elif args.method == 'abs':
        dist = get_spec_step(data_file, round(args.value / dt / dis))
    elif args.method == 'step':
        dist = get_spec_step(data_file, round(args.value / dis))
    elif args.method == 'num':
        dist = get_spec_step(data_file, round(args.value))
    else:
        raise Exception("Unknown method")

    fs = cwd / ("dist" + f"{args.value}_{args.suffix}" + ".csv")

    if args.type == 'win':
        dist_buff: npt.NDArray[np.float32] = np.array([0, 0], dtype=np.float32)
        h = args.h
        i = 0
        while i < cut:
            val = np.sum(dist[i:i+h]) / h / son[cs.fields.volume]
            dist_buff = np.vstack([dist_buff, (i, val)])
            i += h
            h += args.dh
        pd.DataFrame(dist_buff[1:, :], dtype=np.float32).to_csv(fs, header=False, index=False)
    elif args.type == 'norm':
        pd.DataFrame(np.vstack([sizes, dist])).to_csv(fs, header=False, index=False)
    else:
        raise Exception(f"Unknown type: {args.type}")

    return 0


def run(data_file: Path, outfile: Path, temp_file: Path, son: Dict, eps: float, kmin: Union[int, None] = None):
    # dt: float = son[cs.fields.time_step]
    dis: int = son[cs.fields.every]
    Natoms: int = son[cs.fields.N_atoms]
    nvz: float = Natoms/son[cs.fields.volume]

    cut: int = ncut(data_file)
    sizes: npt.NDArray[np.uint64] = np.arange(1, cut + 1, 1, dtype=np.int64)

    temperatures_mat = pd.read_csv(temp_file, header=None)
    temptime: npt.NDArray[np.uint64] = temperatures_mat[0].to_numpy(dtype=np.uint64)
    temperatures: npt.NDArray[np.float32] = temperatures_mat[1].to_numpy(dtype=np.float32)
    # print(f"Temptime shape: {temptime.shape}")
    # print(f"Temperature shape: {temperatures.shape}")

    # N_atoms: int = son[cs.fields.N_atoms]
    if kmin is None:
        Stemp = props.nvs_reverse(nvz)
        Sstep_var: float = temptime[np.argmin(np.abs(temperatures - Stemp))]
        Sstep: int = int(Sstep_var) + 1

        Sdist = get_spec_step(data_file, Sstep)
        Sdist = Sdist * sizes

        kmin = 0
        for i in range(len(Sdist)):
            dat: int = int(np.sum(Sdist[:i]))
            if dat >= eps * Natoms:
                kmin = i + 1
                break
    else:
        pass

    print(f"kmin is {kmin}")

    return proceed(data_file, outfile, son, (temptime, temperatures), cut, kmin)


def main(a: None = None):

    parser = argparse.ArgumentParser(description='Process some floats.')
    parser.add_argument('--debug', action='store_true', help='Debug, prints only parsed arguments')
    # parser.add_argument('--file', action='store', type=str, default=None, required=False, help='File to proceed')
    sub_parsers = parser.add_subparsers(help='Select actrion', dest="command")

    parser_run = sub_parsers.add_parser('run', help='Proceed distribution matrix')
    parser_run.add_argument('--eps', action='store', type=float, default=0.95, required=False, help='Epsilon')
    parser_run.add_argument('--kmin', action='store', type=int, default=None, required=False, help='kmin')
    parser_run.add_argument('--temp_file', action='store', type=str, required=False, help='File with temperatures')
    parser_run.add_argument('--data_file', action='store', type=str, required=False, help='File with data')
    # parser_run.add_argument('--out_file', action='store', type=str, required=False, help='File write to')

    parser_dist = sub_parsers.add_parser('dist', help='Get specific distribution')
    parser_dist.add_argument('--type', action='store', type=str, default='norm', required=False, help='File to proceed')
    parser_dist.add_argument('--h', action='store', type=int, default=1, required=False, help='Window width')
    parser_dist.add_argument('--dh', action='store', type=int, default=0, required=False, help='Window width increment')
    parser_dist.add_argument('--suffix', action='store', type=str, default='', required=False, help='Suffix to filename')

    dist_sub_parsers = parser_dist.add_subparsers(help='Designation method', dest="method")
    parser_name = dist_sub_parsers.add_parser('name', help='Get by name')
    parser_name.add_argument('value', action='store', type=str, default=None, help='Name of moment')
    parser_abs = dist_sub_parsers.add_parser('abs', help='Get by absolute MD time')
    parser_abs.add_argument('value', action='store', type=int, default=None, help='Absolute MD time')

    parser_step = dist_sub_parsers.add_parser('step', help='Get by MD step')
    parser_step.add_argument('value', action='store', type=int, default=None, help='MD step')

    parser_num = dist_sub_parsers.add_parser('num', help='Get by number in distribution matrix')
    parser_num.add_argument('value', action='store', type=int, default=None, help='Number of moment in distribution matrix')

    args: argparse.Namespace = parser.parse_args()

    if args.debug:
        print("Envolved args:")
        print(args)

    cwd = Path.cwd()
    conf_file = cwd / cs.files.data

    with conf_file.open('r') as f:
        son = json.load(f)

    # subf: str = son[cs.fields.data_processing_folder]
    # dt: float = son[cs.fields.time_step]
    # dis: int = son[cs.fields.every]
    # Natoms: int = son[cs.fields.N_atoms]
    km_eps = args.eps
    kmin = args.kmin
    subf: str = son[cs.fields.data_processing_folder]
    data_file: Path = cwd / subf / cs.files.cluster_distribution_matrix if args.data_file is None else Path(args.data_file)
    outfile: Path = cwd / subf / cs.files.comp_data if args.data_file is None else Path(*(list(Path(args.data_file).parts[:-1]) + [cs.files.comp_data]))
    temp_file = cwd / cs.files.temperature if args.temp_file is None else args.temp_file

    # data_file: Path = find_file(cwd, args.file, subf, cs.files.cluster_distribution_matrix)

    if args.command == 'run':
        return run(data_file, outfile, temp_file, son,  km_eps, kmin)
    elif args.command == 'dist':
        raise NotImplementedError
        # return dist_getter(cwd, args, son, data_file, dt, dis, sizes, cut)
    else:
        raise Exception(f"Unknown command: {args.command}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
