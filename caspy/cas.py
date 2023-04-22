# cas.py - Entry point for conjunction analysis algorithm(s).
# Copyright (C) 2021-2023 University of Texas
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
from datetime import datetime, timedelta, timezone
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
            extra_keys=[], window=24.0, cache_data=False):

    pri_files = list(os.path.realpath(os.path.expanduser(f)) for f in pri_files)
    pri_files.sort()
    sec_files = list(os.path.realpath(os.path.expanduser(f)) for f in sec_files)
    sec_files.sort()

    try:
        multiprocessing.set_start_method("spawn")
    except Exception as _:
        pass

    chunk_size, done, pri_task, sec_task, mp_input, summary = os.cpu_count(), set(), [], [], [], []
    pool = multiprocessing.Pool(chunk_size, sms.init_process, (output_path, distance, radius, pos_sigma, vel_sigma, extra_keys, window))

    def build_cache(files1, files2):
        eph_start, eph_stop = 1.0E12, -1.0E12
        all_files = list(set(files1 + files2))
        oem_data = pool.map(import_oem, all_files)
        oem_cache = {all_files[i]:d for i, d in enumerate(oem_data) if (d and len(d[2]) > 5)}

        for f1 in files1:
            val = oem_cache.get(f1)
            if (val):
                eph_start = min(val[1][0], eph_start)
                eph_stop = max(val[1][-1], eph_stop)

        inter_state = pool.map(interpolate, zip(oem_cache.values(), (eph_start,)*len(oem_cache), (eph_stop,)*len(oem_cache)))
        for oc, ix in zip(oem_cache.values(), inter_state):
            oc[1], oc[2] = ix

        return(oem_cache)

    if (cache_data):
        oem_cache = build_cache(pri_files, sec_files)

    for pri_idx, pri_file in enumerate(pri_files):
        done.add(pri_file)
        for sec_idx, sec_file in enumerate(sec_files):
            if (sec_file not in done):
                pri_task.append(pri_file)
                sec_task.append(sec_file)

            if (len(pri_task) == chunk_size or (pri_task and pri_idx == len(pri_files) - 1 and sec_idx == len(sec_files) - 1)):
                if (not cache_data):
                    oem_cache = build_cache(pri_task, sec_task)

                for pt, st in zip(pri_task, sec_task):
                    pri, sec = oem_cache[pt], oem_cache[st]
                    if (not (pri[1] and pri[2] and sec[1] and sec[2])):
                        continue

                    lb, ub = max(pri[1][0], sec[1][0]), min(pri[1][-1], sec[1][-1])
                    p0, p1 = bisect.bisect_left(pri[1], lb), bisect.bisect_right(pri[1], ub)
                    s0, s1 = bisect.bisect_left(sec[1], lb), bisect.bisect_right(sec[1], ub)
                    ut = pri[1][p0:p1]
                    if (len(ut) <= 5):
                        continue

                    pri_map = {"headers": pri[0], "states": pri[2][p0:p1], "covTime": pri[3], "cov": pri[4]}
                    sec_map = {"headers": sec[0], "states": sec[2][s0:s1], "covTime": sec[3], "cov": sec[4]}
                    mp_input.append((pri_map, sec_map, ut))

                if (mp_input):
                    for result in pool.imap_unordered(sms.screen_pair, mp_input):
                        if (result):
                            summary.extend(result)

                pri_task.clear()
                sec_task.clear()
                mp_input.clear()

    pool.close()

    with open(os.path.join(output_path, f"""ca_{datetime.now().strftime("%Y%m%dT%H%M%S")}.txt"""), "w") as fp:
        summary.sort(key=lambda s: s[4])
        for entry in summary:
            fp.write(",".join(str(s) for s in entry) + "\n")

    return(summary)

def run_tle_cas(pri_tles, sec_tles, output_path=".", distance=5000.0, radius=15.0, pos_sigma=1000.0, vel_sigma=0.1,
                window=24.0, cache_data=False):

    pri_tles = sorted(pri_tles, key=lambda k:int(k[1][2:7].strip()))
    sec_tles = sorted(sec_tles, key=lambda k:int(k[1][2:7].strip()))

    try:
        multiprocessing.set_start_method("spawn")
    except Exception as _:
        pass

    chunk_size, done, pri_task, sec_task, mp_input, summary = os.cpu_count(), set(), [], [], [], []
    pool = multiprocessing.Pool(chunk_size, sms.init_process, (output_path, distance, radius, pos_sigma, vel_sigma, [], window))

    def build_cache(tle1, tle2):
        all_tles = tle1 + tle2
        dt0 = datetime.strptime(all_tles[0][1][18:23], "%y%j").replace(tzinfo=timezone.utc) + timedelta(days=float(all_tles[0][1][23:32]))
        t0, t1 = dt0.isoformat(), (dt0 + timedelta(hours=window)).isoformat()

        tle_data = pool.map(propagate_tle, zip(all_tles, (t0,)*len(all_tles), (t1,)*len(all_tles)))
        return({all_tles[i][1]:d for i, d in enumerate(tle_data) if (d and len(d[2]) > 5)})

    if (cache_data):
        tle_cache = build_cache(pri_tles, sec_tles)

    for pri_idx, pri_tle in enumerate(pri_tles):
        done.add(pri_tle[1])
        for sec_idx, sec_tle in enumerate(sec_tles):
            if (sec_tle[1] not in done):
                pri_task.append(pri_tle)
                sec_task.append(sec_tle)

            if (len(pri_task) == chunk_size or (pri_task and pri_idx == len(pri_tles) - 1 and sec_idx == len(sec_tles) - 1)):
                if (not cache_data):
                    tle_cache = build_cache(pri_task, sec_task)

                for pt, st in zip(pri_task, sec_task):
                    pri, sec = tle_cache[pt[1]], tle_cache[st[1]]
                    pri_map = {"headers": pri[0], "states": pri[2], "covTime": [], "cov": []}
                    sec_map = {"headers": sec[0], "states": sec[2], "covTime": [], "cov": []}
                    mp_input.append((pri_map, sec_map, pri[1]))

                if (mp_input):
                    for result in pool.imap_unordered(sms.screen_pair, mp_input):
                        if (result):
                            summary.extend(result)

                pri_task.clear()
                sec_task.clear()
                mp_input.clear()

    pool.close()

    with open(os.path.join(output_path, f"""ca_{datetime.now().strftime("%Y%m%dT%H%M%S")}.txt"""), "w") as fp:
        summary.sort(key=lambda s: s[4])
        for entry in summary:
            fp.write(",".join(str(s) for s in entry) + "\n")

    return(summary)

def basic_screen(pri, sec, start_utc, stop_utc):
    try:
        multiprocessing.set_start_method("spawn")
    except Exception as _:
        pass

    pool = multiprocessing.Pool(2, sms.init_process, (None, None, None, None, None, [], 366*24))

    if (isinstance(pri, str) and os.path.isfile(pri) and isinstance(sec, str) and os.path.isfile(sec)):
        t0, t1 = get_J2000_epoch_offset((start_utc, stop_utc))
        oem_data = pool.map(import_oem, (pri, sec))
        interp = pool.map(interpolate, zip(oem_data, (t0, t0), (t1, t1)))
        results = zip(interp[0][0], interp[0][1], interp[1][1])
    else:
        prop = pool.map(propagate_tle, zip((pri, sec), (start_utc, start_utc), (stop_utc, stop_utc)))
        results = zip(prop[0][1], prop[0][2], prop[1][2])

    pool.close()
    norm = lambda x, y: (np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2),
                         np.sqrt((x[3] - y[3])**2 + (x[4] - y[4])**2 + (x[5] - y[5])**2))

    return([[get_UTC_string(t, precision=6), s1, s2, *norm(s1, s2)] for t, s1, s2 in results])

def import_oem(oem_file):
    try:
        headers = {"EPHEMERIS_NAME": os.path.basename(oem_file)}
        times, states, cov_start, cov_times, cov, end_time = [], [], False, [], [], None
        with open(oem_file, "r") as fp:
            lines = [l for l in fp.read().splitlines() if (l)]

        for idx, line in enumerate(lines):
            # Import extra key/value pairs from comment lines
            if (line.startswith("COMMENT")):
                toks = line.split()
                if (len(toks) > 2 and toks[1] in sms._extra_keys):
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
                                cov[-1].extend(float(t)*1.0E6 for t in lines[i].split())
                            if (len(cov[-1]) == 21):
                                break
                continue

            # Import headers and states
            if ("=" in line):
                toks = [t.strip() for t in line.split("=")]
                headers[toks[0]] = toks[1]
            elif (":" in line and line[0].isnumeric()):
                toks = line.split()
                if (not end_time):
                    end_time = (datetime.fromisoformat(toks[0]).replace(tzinfo=timezone.utc) +
                                timedelta(hours=sms._window)).isoformat(timespec="milliseconds")

                curr_time = datetime.fromisoformat(toks[0]).replace(tzinfo=timezone.utc)
                if (toks[0] <= end_time):
                    times.append(toks[0])
                    states.append([float(t)*1000.0 for t in toks[1:]])

        times = get_J2000_epoch_offset(times)
    except Exception as exc:
        print(f"{oem_file}: {exc}")
        return(None)

    return([headers, [times] if (isinstance(times, float)) else list(times), states, cov_times, cov])

def interpolate(p):
    try:
        if (p[2] - p[1] > 1.0E-6):
            ut = np.arange(p[1], p[2], 60.0).tolist()
            if (ut[-1] != p[2]):
                ut.append(p[2])
        else:
            ut = [p[2]]

        i0, i1 = bisect.bisect_left(ut, p[0][1][0]), bisect.bisect_right(ut, p[0][1][-1])
        ixt = ut[i0:i1]

        ixs = [i.true_state[:] for i in interpolate_ephemeris(Frame.EME2000, p[0][1], p[0][2], Frame.EME2000, ixt, 0.0, 0.0, interp_method=1)]
    except Exception as exc:
        ixt, ixs = [], []
        print(f"interpolate_ephemeris error: {exc}")

    return(ixt, ixs)

def propagate_tle(params):
    try:
        tle, t0, t1 = params
        tt = get_J2000_epoch_offset(params[1:3])

        config = [configure(prop_initial_TLE=tle[1:], prop_start=tt[0], prop_step=60.0, prop_end=tt[1], prop_inertial_frame=Frame.EME2000,
                            gravity_degree=-1, gravity_order=-1, ocean_tides_degree=-1, ocean_tides_order=-1, third_body_sun=False,
                            third_body_moon=False, solid_tides_sun=False, solid_tides_moon=False, drag_model=DragModel.UNDEFINED, rp_sun=False)]
        headers = {"OBJECT_ID": str(int(tle[1][2:7])), "OBJECT_NAME": tle[0][2:].strip(), "START_TIME": t0, "STOP_TIME": t1,
                   "EPHEMERIS_NAME": "TLE"}

        times, states = [], []
        for p in propagate_orbits(config)[0].array:
            times.append(p.time)
            states.append(list(p.true_state))
    except Exception as exc:
        print(f"{tle[1:]}: {exc}")
        return(None)

    return(headers, times, states)

def dir_or_file(param):
    if (os.path.isdir(param) or os.path.isfile(param)):
        return(param)

    raise(argparse.ArgumentTypeError(f"{param} is not a valid directory or file"))

def list_oem_files(path_or_file):
    if (os.path.isdir(path_or_file)):
        return([f for f in glob.iglob(os.path.join(path_or_file, "**"), recursive=True) if (os.path.isfile(f) and f.endswith(".oem"))])
    else:
        return([path_or_file])

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="Conjunction Analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("primary-path", help="Primary object OEM file or directory path", type=dir_or_file)
    parser.add_argument("secondary-path", help="Secondary object OEM file or directory path", type=dir_or_file)
    parser.add_argument("-o", "--output-path", help="CDM output path", default=".")
    parser.add_argument("-d", "--distance", help="Critical distance [m]", type=float, default=5000.0)
    parser.add_argument("-r", "--radius", help="Hard body radius [m]", type=float, default=15.0)
    parser.add_argument("-p", "--pos-sigma", help="Default position one-sigma [m]", type=float, default=1000.0)
    parser.add_argument("-v", "--vel-sigma", help="Default velocity one-sigma [m/s]", type=float, default=0.1)
    parser.add_argument("-e", "--extra-keys", help="Extra COMMENT key/value data to copy", type=str, default="")
    parser.add_argument("-w", "--window", help="Screening time window [hr]", type=float, default=24.0)
    parser.add_argument("-c", "--cache-data", help="Cache all ephemeris data in memory for speed", action="store_true", default=False)

    if (len(sys.argv) == 1):
        parser.print_help()
        exit(1)

    start_time = datetime.utcnow()
    arg = parser.parse_args()
    pri_files = list_oem_files(getattr(arg, "primary-path"))
    sec_files = list_oem_files(getattr(arg, "secondary-path"))

    run_cas(pri_files, sec_files, output_path=arg.output_path, distance=arg.distance, radius=arg.radius, pos_sigma=arg.pos_sigma,
            vel_sigma=arg.vel_sigma, extra_keys=arg.extra_keys.split(","), window=arg.window, cache_data=arg.cache_data)

    tmin, tsec = divmod((datetime.utcnow() - start_time).total_seconds(), 60.0)
    print(f"Elapsed time = {tmin:.0f} min {tsec:.1f} sec")
