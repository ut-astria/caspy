# caspy.py - Wrapper for conjunction analysis algorithm(s).
# Copyright (C) 2021-2022 University of Texas
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
from smart_sieve import init_process, screen_pair
from datetime import datetime
import glob
import itertools
import multiprocessing
import numpy as np
from orbdetpy import Frame
from orbdetpy.conversion import get_J2000_epoch_offset
from orbdetpy.utilities import interpolate_ephemeris
import os
import sys
import tqdm

def isdir(path):
    if (os.path.isdir(path)):
        return(path)
    else:
        raise(argparse.ArgumentTypeError(f"Invalid path {path}"))

def import_oem(oem_file):
    with open(oem_file, "r") as fp:
        lines = [l.strip() for l in fp.readlines() if (l.strip())]

    headers, times, states = {}, [], []
    for line in lines:
        if (line.startswith("COVARIANCE_START")):
            break
        if ("=" in line):
            toks = [t.strip() for t in line.split("=")]
            headers[toks[0]] = toks[1]
        elif (line[0].isnumeric() and ":" in line):
            toks = line.split()
            times.append(toks[0])
            states.append([float(t)*1000.0 for t in toks[1:]])
    return(oem_file, headers, list(get_J2000_epoch_offset(times)), states)

def interpolate(params):
    return([list(ixs.true_state) for ixs in interpolate_ephemeris(*params)])

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="Conjunction Analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("primary-path", help="Primary object OEM file path", type=isdir)
    parser.add_argument("secondary-path", help="Secondary object OEM file path", type=isdir)
    parser.add_argument("-o", "--output-path", help="CDM output path", type=isdir, default=os.path.join("."))
    parser.add_argument("-d", "--distance", help="Critical distance [m]", type=float, default=5000.0)
    parser.add_argument("-r", "--radius", help="Hard body radius [m]", type=float, default=15.0)
    parser.add_argument("-p", "--pos-sigma", help="Position standard deviation [m]", type=float, default=1000.0)
    parser.add_argument("-v", "--vel-sigma", help="Velocity standard deviation [m/s]", type=float, default=0.1)
    parser.add_argument("-n", "--inter-order", help="Ephemeris interpolation order", type=int, default=5)
    parser.add_argument("-t", "--inter-time", help="Interpolation time [s]", type=float, default=0.01)
    if (len(sys.argv) == 1):
        parser.print_help()
        exit(1)

    multiprocessing.set_start_method("spawn")
    pool = multiprocessing.Pool(os.cpu_count(), init_process)

    args = parser.parse_args()
    pri_files = glob.glob(os.path.join(getattr(args, "primary-path"), "*.oem"))
    sec_files = glob.glob(os.path.join(getattr(args, "secondary-path"), "*.oem"))
    print(f"{len(pri_files)} primaries x {len(sec_files)} secondaries / {os.cpu_count()}"
          f" CPU cores = {int(np.ceil(len(pri_files)*len(sec_files)/os.cpu_count()))} tasks")

    # Object 1 (primary) loop
    primary = []
    for fname, headers, pri_time, pri_state in pool.map(import_oem, pri_files):
        primary.append({"objName": headers["OBJECT_NAME"], "objID": headers["OBJECT_ID"], "startTime": headers["START_TIME"],
                        "endTime": headers["STOP_TIME"], "times": pri_time, "states": pri_state, "oemFile": fname})

    # Process primary/secondary combinations in parallel chunks based on CPU core count
    tasks, summary = list(itertools.product(primary, sec_files)), []
    for task in tqdm.tqdm([tasks[i:i + os.cpu_count()] for i in range(0, len(tasks), os.cpu_count())],
                          desc="Tasks", unit="task", disable=None):
        mp_inputs = []
        for fname, headers, sec_time, sec_state in pool.map(import_oem, [t[-1] for t in task]):
            sec_data = {"objName": headers["OBJECT_NAME"], "objID": headers["OBJECT_ID"], "startTime": headers["START_TIME"],
                        "endTime": headers["STOP_TIME"], "oemFile": fname}
            pri = task[len(mp_inputs)][0].copy()
            inter_time = np.arange(max(pri["times"][0], sec_time[0]), min(pri["times"][-1], sec_time[-1]), 60.0).tolist()
            params = ((Frame.EME2000, pri["times"], pri["states"], args.inter_order, Frame.EME2000, inter_time, 0.0, 0.0),
                      (Frame.EME2000, sec_time, sec_state, args.inter_order, Frame.EME2000, inter_time, 0.0, 0.0))
            pri["states"], sec_data["states"] = pool.map(interpolate, params)
            mp_inputs.append([pri, sec_data, inter_time, args.output_path, args.inter_order, args.inter_time,
                              args.distance, args.pos_sigma, args.vel_sigma, args.radius])

        for result in pool.imap_unordered(screen_pair, mp_inputs):
            if (result):
                summary.extend(result)

    pool.close()
    summary.sort(key=lambda s: s[4])
    with open(os.path.join(args.output_path, f"""ca-{datetime.now().strftime("%Y%m%dT%H%M%S")}.txt"""), "w") as fp:
        for entry in summary:
            fp.write(" ".join(str(s) for s in entry) + "\n")
