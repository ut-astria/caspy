# cas.py - Entry point for conjunction analysis algorithm(s).
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
if (__name__ == "caspy.cas"):
    import caspy.smart_sieve as sms
else:
    import smart_sieve as sms

def run_cas(pri_files, sec_files, output_path=".", distance=5000.0, radius=15.0, pos_sigma=1000.0, vel_sigma=0.1,
            inter_order=5, inter_time=0.01, extra_keys=[]):
    multiprocessing.set_start_method("spawn")
    pool = multiprocessing.Pool(os.cpu_count(), sms.init_process)

    # Object 1 (primary) loop
    primary = []
    for fname, headers, pri_time, pri_state, cov_time, cov in pool.map(import_oem, zip(pri_files, (extra_keys,)*len(pri_files))):
        primary.append({"headers": headers, "times": pri_time, "states": pri_state, "oemFile": fname, "covTime": cov_time, "cov": cov})

    # Process primary/secondary combinations in parallel chunks depending on CPUs
    tasks, summary = list(itertools.product(primary, sec_files)), []
    for task in [tasks[i:i + os.cpu_count()] for i in range(0, len(tasks), os.cpu_count())]:
        mp_inputs = []
        for fname, headers, sec_time, sec_state, cov_time, cov in pool.map(
                import_oem, zip((t[-1] for t in task), (extra_keys,)*os.cpu_count())):
            sec_data = {"headers": headers, "oemFile": fname, "covTime": cov_time, "cov": cov}
            pri = task[len(mp_inputs)][0].copy()
            exp_time = np.arange(max(pri["times"][0], sec_time[0]), min(pri["times"][-1], sec_time[-1]), 60.0).tolist()
            params = ((Frame.EME2000, pri["times"], pri["states"], inter_order, Frame.EME2000, exp_time, 0.0, 0.0),
                      (Frame.EME2000, sec_time, sec_state, inter_order, Frame.EME2000, exp_time, 0.0, 0.0))
            pri["states"], sec_data["states"] = pool.map(interpolate, params)
            mp_inputs.append([pri, sec_data, exp_time, output_path, inter_order, inter_time, distance, pos_sigma, vel_sigma, radius])

        for result in pool.imap_unordered(sms.screen_pair, mp_inputs):
            if (result):
                summary.extend(result)

    pool.close()
    summary.sort(key=lambda s: s[4])
    with open(os.path.join(output_path, f"""ca-{datetime.now().strftime("%Y%m%dT%H%M%S")}.txt"""), "w") as fp:
        for entry in summary:
            fp.write(",".join(str(s) for s in entry) + "\n")

def import_oem(params):
    oem_file, extra_keys = params
    with open(oem_file, "r") as fp:
        lines = [l.strip() for l in fp.readlines() if (l.strip())]

    headers, times, states, cov_start, cov_times, cov = {}, [], [], False, [], []
    for idx, line in enumerate(lines):
        # Import extra key/value pairs from comment lines
        if (line.startswith("COMMENT")):
            toks = line.split()
            if (len(toks) > 2 and toks[1] in extra_keys):
                headers.setdefault("extra", {})[toks[1]] = " ".join(toks[2:])
            continue

        # Import covariance
        if (line.startswith("COVARIANCE_START")):
            cov_start = True
        if (line.startswith("COVARIANCE_STOP")):
            cov_times = list(get_J2000_epoch_offset(cov_times))
            break
        if (cov_start):
            if (line.startswith("EPOCH")):
                cov.append([])
                cov_times.append(line.split("=")[-1].strip())
                for i in range(idx + 1, len(lines)):
                    if not ("=" in lines[i] or lines[i].startswith("COMMENT")):
                        cov[-1].extend(float(t)*1E6 for t in lines[i].split())
                    if (len(cov[-1]) == 21):
                        break
            continue

        # Import headers and states
        if ("=" in line):
            toks = [t.strip() for t in line.split("=")]
            headers[toks[0]] = toks[1]
        elif (line[0].isnumeric() and ":" in line):
            toks = line.split()
            times.append(toks[0])
            states.append([float(t)*1000.0 for t in toks[1:]])
    return(oem_file, headers, list(get_J2000_epoch_offset(times)), states, cov_times, cov)

def interpolate(params):
    return([list(ixs.true_state) for ixs in interpolate_ephemeris(*params)])

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="Conjunction Analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("primary-path", help="Primary object OEM file path")
    parser.add_argument("secondary-path", help="Secondary object OEM file path")
    parser.add_argument("-o", "--output-path", help="CDM output path", default=".")
    parser.add_argument("-d", "--distance", help="Critical distance [m]", type=float, default=5000.0)
    parser.add_argument("-r", "--radius", help="Hard body radius [m]", type=float, default=15.0)
    parser.add_argument("-p", "--pos-sigma", help="Position standard deviation [m]", type=float, default=1000.0)
    parser.add_argument("-v", "--vel-sigma", help="Velocity standard deviation [m/s]", type=float, default=0.1)
    parser.add_argument("-n", "--inter-order", help="Ephemeris interpolation order", type=int, default=5)
    parser.add_argument("-t", "--inter-time", help="Interpolation time [s]", type=float, default=0.01)
    parser.add_argument("-e", "--extra-keys", help="Extra COMMENT key/value data to copy", type=str, default="")
    if (len(sys.argv) == 1):
        parser.print_help()
        exit(1)

    arg = parser.parse_args()
    pri = glob.glob(os.path.join(getattr(arg, "primary-path"), "*.oem"))
    sec = glob.glob(os.path.join(getattr(arg, "secondary-path"), "*.oem"))
    run_cas(pri, sec, arg.output_path, arg.distance, arg.radius, arg.pos_sigma, arg.vel_sigma,
            arg.inter_order, arg.inter_time, arg.extra_keys.split(","))
