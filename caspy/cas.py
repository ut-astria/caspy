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
import bisect
from datetime import datetime, timedelta
import glob
import multiprocessing
import numpy as np
from orbdetpy import configure, DragModel, Frame
from orbdetpy.conversion import get_J2000_epoch_offset, get_UTC_string
from orbdetpy.propagation import propagate_orbits
from orbdetpy.utilities import interpolate_ephemeris
import os
import sys

if (__name__ == "caspy.cas"):
    import caspy.smart_sieve as sms
else:
    import smart_sieve as sms

def run_cas(pri_files, sec_files, output_path=".", distance=5000.0, radius=15.0, pos_sigma=1000.0, vel_sigma=0.1,
            inter_order=5, inter_time=0.01, extra_keys=[], window=24.0, chunk_size=os.cpu_count()):
    if (not hasattr(pri_files, "__len__")):
        pri_files = list(pri_files)
    if (not hasattr(sec_files, "__len__")):
        sec_files = list(sec_files)

    try:
        multiprocessing.set_start_method("spawn")
    except Exception as _:
        pass
    pool = multiprocessing.Pool(chunk_size, sms.init_process)
    done, task_cache, pri_task, sec_task, mp_input, summary = set(), [], [], [], [], []

    for pri_idx, pri_file in enumerate(pri_files):
        done.add(pri_file)
        for sec_idx, sec_file in enumerate(sec_files):
            if (sec_file not in done):
                pri_task.append(pri_file)
                sec_task.append(sec_file)

            if (len(pri_task) == chunk_size or (pri_task and pri_idx == len(pri_files) - 1 and sec_idx == len(sec_files) - 1)):
                secondary = pool.map(import_oem, zip(sec_task, (extra_keys,)*len(sec_task), (window,)*len(sec_task)))
                if (any(not s for s in secondary)):
                    pri_task.clear()
                    sec_task.clear()
                    continue

                if (pri_task != task_cache):
                    primary = pool.map(import_oem, zip(pri_task, (extra_keys,)*len(pri_task), (window,)*len(pri_task)))
                    if (any(not p for p in primary)):
                        pri_task.clear()
                        sec_task.clear()
                        continue

                    pri_time = [np.arange(p[2][0], p[2][-1], 180.0).tolist() for p in primary]
                    pri_state = pool.map(interpolate, ((Frame.EME2000, p[2], p[3], inter_order, Frame.EME2000, e, 0.0, 0.0)
                                                       for e, p in zip(pri_time, primary)))
                    task_cache = pri_task.copy()

                for pri, sec, ptm, pst in zip(primary, secondary, pri_time, pri_state):
                    idx0, idx1 = bisect.bisect_left(ptm, sec[2][0]), bisect.bisect_right(ptm, sec[2][-1])
                    use_time, use_state = ptm[idx0:idx1], pst[idx0:idx1]
                    if (len(use_time) < 2):
                        continue
                    sec_state = interpolate((Frame.EME2000, sec[2], sec[3], inter_order, Frame.EME2000, use_time, 0.0, 0.0))

                    pri_map = {"oemFile": pri[0], "headers": pri[1], "states": use_state, "covTime": pri[4], "cov": pri[5]}
                    sec_map = {"oemFile": sec[0], "headers": sec[1], "states": sec_state, "covTime": sec[4], "cov": sec[5]}
                    mp_input.append([pri_map, sec_map, use_time, output_path, inter_order, inter_time, distance, pos_sigma, vel_sigma, radius])

                if (mp_input):
                    for result in pool.imap_unordered(sms.screen_pair, mp_input):
                        if (result):
                            summary.extend(result)

                pri_task.clear()
                sec_task.clear()
                mp_input.clear()

    pool.close()
    summary.sort(key=lambda s: s[4])
    with open(os.path.join(output_path, f"""ca_{datetime.now().strftime("%Y%m%dT%H%M%S")}.txt"""), "w") as fp:
        for entry in summary:
            fp.write(",".join(str(s) for s in entry) + "\n")

def run_tle_cas(pri_tles, sec_tles, output_path=".", distance=5000.0, radius=15.0, pos_sigma=1000.0, vel_sigma=0.1,
                inter_order=5, inter_time=0.01, window=24.0, chunk_size=os.cpu_count()):
    if (not hasattr(pri_tles, "__len__")):
        pri_tles = list(pri_tles)
    if (not hasattr(sec_tles, "__len__")):
        sec_tles = list(sec_tles)

    try:
        multiprocessing.set_start_method("spawn")
    except Exception as _:
        pass
    pool = multiprocessing.Pool(chunk_size, sms.init_process)
    done, task_cache, pri_task, sec_task, mp_input, summary = set(), [], [], [], [], []

    for pri_idx, pri_tle in enumerate(pri_tles):
        done.add(pri_tle[1])
        for sec_idx, sec_tle in enumerate(sec_tles):
            if (sec_tle[1] not in done):
                pri_task.append(pri_tle)
                sec_task.append(sec_tle)

            if (len(pri_task) == chunk_size or (pri_task and pri_idx == len(pri_tles) - 1 and sec_idx == len(sec_tles) - 1)):
                if (pri_task != task_cache):
                    primary = pool.map(propagate_tle, zip(pri_task, (window,)*len(pri_task), (None,)*len(pri_task), (None,)*len(pri_task)))
                    if (any(not p for p in primary)):
                        pri_task.clear()
                        sec_task.clear()
                        continue
                    task_cache = pri_task.copy()

                secondary = pool.map(propagate_tle,
                                     zip(sec_task, (window,)*len(pri_task), (p[2][0] for p in primary), (p[2][-1] for p in primary)))
                if (any(not s for s in secondary)):
                    pri_task.clear()
                    sec_task.clear()
                    continue

                for pri, sec in zip(primary, secondary):
                    pri_map = {"oemFile": pri[0], "headers": pri[1], "states": pri[3], "covTime": [], "cov": []}
                    sec_map = {"oemFile": sec[0], "headers": sec[1], "states": sec[3], "covTime": [], "cov": []}
                    mp_input.append([pri_map, sec_map, pri[2], output_path, inter_order, inter_time, distance, pos_sigma, vel_sigma, radius])

                if (mp_input):
                    for result in pool.imap_unordered(sms.screen_pair, mp_input):
                        if (result):
                            summary.extend(result)

                pri_task.clear()
                sec_task.clear()
                mp_input.clear()

    pool.close()
    summary.sort(key=lambda s: s[4])
    with open(os.path.join(output_path, f"""tle_ca_{datetime.now().strftime("%Y%m%dT%H%M%S")}.txt"""), "w") as fp:
        for entry in summary:
            fp.write(",".join(str(s) for s in entry) + "\n")

def import_oem(params):
    try:
        oem_file, extra_keys, window = params
        with open(oem_file, "r") as fp:
            lines = [l.strip() for l in fp.readlines() if (l.strip())]

        headers, times, states, cov_start, cov_times, cov, end_time = {}, [], [], False, [], [], None
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
                if (cov_times):
                    cov_times = get_J2000_epoch_offset(cov_times)
                    cov_times = [cov_times] if (isinstance(cov_times, float)) else list(cov_times)
                break

            if (cov_start):
                if (line.startswith("EPOCH")):
                    ctime = line.split("=")[-1].strip()
                    if (ctime <= end_time):
                        cov.append([])
                        cov_times.append(ctime)
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
                if (not end_time):
                    end_time = (datetime.fromisoformat(toks[0]) + timedelta(hours=window)).isoformat(timespec="milliseconds")
                if (toks[0] <= end_time):
                    times.append(toks[0])
                    states.append([float(t)*1000.0 for t in toks[1:]])

        times = get_J2000_epoch_offset(times)
    except Exception as exc:
        print(f"{oem_file}: {exc}")
        return(None)

    return(oem_file, headers, [times] if (isinstance(times, float)) else list(times), states, cov_times, cov)

def propagate_tle(params):
    try:
        t0t1, times, states = [None]*2, [], []
        tle, window, t0t1[0], t0t1[1] = params
        if (t0t1[0] and t0t1[1]):
            tint = get_UTC_string(t0t1)
        else:
            tint = ((datetime.strptime(tle[1][18:23], "%y%j") + timedelta(days=float(tle[1][23:32]))).isoformat(),
                    (datetime.strptime(tle[1][18:23], "%y%j") + timedelta(days=float(tle[1][23:32]), hours=window)).isoformat())
            t0t1 = get_J2000_epoch_offset(tint)

        config = [configure(prop_initial_TLE=tle[1:3], prop_start=t0t1[0], prop_step=180.0, prop_end=t0t1[1], prop_inertial_frame=Frame.EME2000,
                            gravity_degree=-1, gravity_order=-1, ocean_tides_degree=-1, ocean_tides_order=-1, third_body_sun=False, third_body_moon=False,
                            solid_tides_sun=False, solid_tides_moon=False, drag_model=DragModel.UNDEFINED, rp_sun=False, sim_measurements=False)]
        headers = {"OBJECT_ID": str(int(tle[1][2:7])), "OBJECT_NAME": tle[0][2:].strip(), "START_TIME": tint[0], "STOP_TIME": tint[1]}

        for p in propagate_orbits(config)[0].array:
            times.append(p.time)
            states.append(list(p.true_state))
    except Exception as exc:
        print(f"{tle[1:3]}: {exc}")
        return(None)

    return(headers["OBJECT_ID"], headers, times, states)

def interpolate(params):
    return([list(ie.true_state) for ie in interpolate_ephemeris(*params)])

def dir_or_file(param):
    if (os.path.isdir(param) or os.path.isfile(param)):
        return(param)
    raise(argparse.ArgumentTypeError(f"{param} is not a valid directory or file"))

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="Conjunction Analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("primary-path", help="Primary object OEM file or directory path", type=dir_or_file)
    parser.add_argument("secondary-path", help="Secondary object OEM file or directory path", type=dir_or_file)
    parser.add_argument("-d", "--distance", help="Critical distance [m]", type=float, default=5000.0)
    parser.add_argument("-w", "--window", help="Screening time window [hr]", type=float, default=24.0)
    parser.add_argument("-o", "--output-path", help="CDM output path", default=".")
    parser.add_argument("-e", "--extra-keys", help="Extra COMMENT key/value data to copy", type=str, default="")
    parser.add_argument("-r", "--radius", help="Hard body radius [m]", type=float, default=15.0)
    parser.add_argument("-p", "--pos-sigma", help="Default position one-sigma [m]", type=float, default=1000.0)
    parser.add_argument("-v", "--vel-sigma", help="Default velocity one-sigma [m/s]", type=float, default=0.1)
    parser.add_argument("-n", "--inter-order", help="Ephemeris interpolation order", type=int, default=5)
    parser.add_argument("-t", "--inter-time", help="Interpolation time [s]", type=float, default=0.01)
    if (len(sys.argv) == 1):
        parser.print_help()
        exit(1)

    arg = parser.parse_args()
    pri_path, sec_path = getattr(arg, "primary-path"), getattr(arg, "secondary-path")
    pri_files = glob.glob(os.path.join(pri_path, "*.oem")) if (os.path.isdir(pri_path)) else [pri_path]
    sec_files = glob.glob(os.path.join(sec_path, "*.oem")) if (os.path.isdir(sec_path)) else [sec_path]
    run_cas(pri_files, sec_files, arg.output_path, arg.distance, arg.radius, arg.pos_sigma, arg.vel_sigma,
            arg.inter_order, arg.inter_time, arg.extra_keys.split(","), arg.window)
