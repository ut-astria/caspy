# smart_sieve.py - Conjunction analysis using the smart sieve algorithm.
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

import bisect
from datetime import datetime, timezone
import json
from mako.template import Template
import numpy as np
import numpy.linalg as la
from orbdetpy import configure, Frame
from orbdetpy.conversion import get_J2000_epoch_offset, get_UTC_string, ltr_to_matrix
from orbdetpy.propagation import propagate_orbits
from orbdetpy.utilities import interpolate_ephemeris
from os import getenv, path

_MU_EARTH = 3.986004415E14

def init_process(*params):
    global _output_path, _critical_dist, _body_radius, _pos_sigma, _vel_sigma, _extra_keys, _window
    _output_path, _critical_dist, _body_radius, _pos_sigma, _vel_sigma, _extra_keys, _window = params

    global _cospar_to_norad, _norad_to_cospar, _debug_mode, _debug_data, _cdm_template
    _cospar_to_norad, _norad_to_cospar = {}, {}
    _debug_mode = getenv("CASPY_DEBUG", "0") == "1"
    _cdm_template = Template(filename=path.join(path.dirname(path.realpath(__file__)), "template.cdm"))

    # Load UT object ID catalog if it exists
    try:
        cat_file = getenv("CASPY_OBJECT_CATALOG", path.expanduser(path.join("~", "object_catalog.csv")))
        with open(cat_file, "r") as fp:
            for line in fp.read().splitlines():
                tok = line.split(",")
                _cospar_to_norad[tok[1]] = tok[0]
                _norad_to_cospar[tok[0]] = tok[1]
    except Exception as _:
        pass

def screen_pair(params):
    global _output_path, _critical_dist, _body_radius, _pos_sigma, _vel_sigma, _extra_keys, _window
    global _cdm_template, _cospar_to_norad, _debug_mode, _debug_data

    try:
        object1, object2, times = params
        states1, states2 = object1["states"], object2["states"]

        # If the OEM file had a COSPAR ID then map it to a NORAD ID else leave it as is
        object1["headers"]["OBJECT_ID"] = _cospar_to_norad.get(object1["headers"]["OBJECT_ID"], object1["headers"]["OBJECT_ID"])
        object2["headers"]["OBJECT_ID"] = _cospar_to_norad.get(object2["headers"]["OBJECT_ID"], object2["headers"]["OBJECT_ID"])

        # Skip cases where primary and secondary have the same object ID
        if (object1["headers"]["OBJECT_ID"] == object2["headers"]["OBJECT_ID"]):
            return(None)

        object1["headers"]["COSPAR_ID"] = _norad_to_cospar.get(object1["headers"]["OBJECT_ID"], object1["headers"]["OBJECT_ID"])
        object2["headers"]["COSPAR_ID"] = _norad_to_cospar.get(object2["headers"]["OBJECT_ID"], object2["headers"]["OBJECT_ID"])

        # Apogee/perigee filter
        apsis_pass, rp1, rp2 = apogee_filter(states1[0], states2[0], times[1] - times[0])
        if (not apsis_pass):
            return(None)

        dt = times[1] - times[0]
        prop = propagate(times[0], states1[0], states2[0], times[0] - 2.0*dt, -dt)
        times.insert(0, prop[0].array[1].time)
        times.insert(0, prop[0].array[2].time)
        states1.insert(0, prop[0].array[1].true_state)
        states1.insert(0, prop[0].array[2].true_state)
        states2.insert(0, prop[1].array[1].true_state)
        states2.insert(0, prop[1].array[2].true_state)

        prop = propagate(times[-1], states1[-1], states2[-1], times[-1] + 2.0*dt, dt)
        times.append(prop[0].array[1].time)
        times.append(prop[0].array[2].time)
        states1.append(prop[0].array[1].true_state)
        states1.append(prop[0].array[2].true_state)
        states2.append(prop[1].array[1].true_state)
        states2.append(prop[1].array[2].true_state)

        index, event_found, event_min_dist, summary = 2, False, _critical_dist, []

        if (_debug_mode):
            _debug_data = {}

        # Iterate through each time step
        while (index < len(times) - 2):
            # Run smart Sieve
            sieve_pass, skip_index = sift(states1[index], states2[index], rp1, rp2, 180.0)
            if (not sieve_pass):
                index += skip_index
                continue

            inter_time = [times[index + j] for j in range(-2, 3)]
            inter_state1 = [states1[index + j] for j in range(-2, 3)]
            inter_state2 = [states2[index + j] for j in range(-2, 3)]
            index += 1

            # If smart sieve passed, run binary search
            tca_result = find_tca(inter_time, inter_state1, inter_state2, times[1] - times[0], rp1, rp2)
            if (not tca_result):
                continue

            # Event found
            if (tca_result[1] < event_min_dist):
                event_found = True
                time_ca, event_min_dist, state_ca1, state_ca2 = tca_result

            # If event has just ended
            if (event_found and (tca_result[1] >= event_min_dist or index == len(times) - 2)):
                ca_utc = get_UTC_string(time_ca, precision=6)
                screen_start, screen_stop = get_UTC_string((times[2], times[-3]))

                pos_obj1, vel_obj1 = state_ca1[:3], state_ca1[3:]
                pos_obj2, vel_obj2 = state_ca2[:3], state_ca2[3:]
                miss_dist = la.norm(np.subtract(pos_obj1, pos_obj2))

                def_cov = [_pos_sigma**2, 0, _pos_sigma**2, 0, 0, _pos_sigma**2, 0, 0, 0, _vel_sigma**2,
                           0, 0, 0, 0, _vel_sigma**2, 0, 0, 0, 0, 0, _vel_sigma**2]
                cov1 = np.array(ltr_to_matrix(get_covariance(object1, time_ca, def_cov)))
                cov2 = np.array(ltr_to_matrix(get_covariance(object2, time_ca, def_cov)))

                rot1_3x3, rot1_6x6 = get_rotation(pos_obj1, vel_obj1)
                rot2_3x3, rot2_6x6 = get_rotation(pos_obj2, vel_obj2)

                cov1_rtn = rot1_6x6.dot(cov1).dot(rot1_6x6.transpose())
                cov2_rtn = rot2_6x6.dot(cov2).dot(rot2_6x6.transpose())
                rel_pos = rot1_3x3.dot(np.subtract(pos_obj2, pos_obj1))
                rel_vel = rot1_3x3.dot(np.subtract(vel_obj2, vel_obj1))
                coll_prob = compute_pc(state_ca1, state_ca2, cov1, cov2)

                object1.update({"CREATION_DATE":datetime.now(timezone.utc).isoformat(timespec="milliseconds")[:-6], "TCA": ca_utc,
                                "MISS_DISTANCE": miss_dist, "RELATIVE_SPEED": la.norm(np.subtract(vel_obj1, vel_obj2)),
                                "RELATIVE_POSITION_R":rel_pos[0],"RELATIVE_POSITION_T":rel_pos[1],"RELATIVE_POSITION_N":rel_pos[2],
                                "RELATIVE_VELOCITY_R":rel_vel[0],"RELATIVE_VELOCITY_T":rel_vel[1],"RELATIVE_VELOCITY_N":rel_vel[2],
                                "START_SCREEN_PERIOD": screen_start, "STOP_SCREEN_PERIOD": screen_stop, "COLLISION_PROBABILITY": coll_prob,
                                "X": pos_obj1[0]/1000, "Y": pos_obj1[1]/1000, "Z": pos_obj1[2]/1000,"X_DOT":vel_obj1[0]/1000,
                                "Y_DOT":vel_obj1[1]/1000, "Z_DOT":vel_obj1[2]/1000,"CR_R":cov1_rtn[0][0],"CT_R":cov1_rtn[1][0],
                                "CT_T":cov1_rtn[1][1],"CN_R":cov1_rtn[2][0],"CN_T":cov1_rtn[2][1],"CN_N":cov1_rtn[2][2],"CRDOT_R":cov1_rtn[3][0],
                                "CRDOT_T":cov1_rtn[3][1],"CRDOT_N":cov1_rtn[3][2],"CRDOT_RDOT":cov1_rtn[3][3],"CTDOT_R":cov1_rtn[4][0],
                                "CTDOT_T":cov1_rtn[4][1],"CTDOT_N":cov1_rtn[4][2],"CTDOT_RDOT":cov1_rtn[4][3],"CTDOT_TDOT":cov1_rtn[4][4],
                                "CNDOT_R":cov1_rtn[5][0],"CNDOT_T":cov1_rtn[5][1],"CNDOT_N":cov1_rtn[5][2],"CNDOT_RDOT":cov1_rtn[5][3],
                                "CNDOT_TDOT": cov1_rtn[5][4], "CNDOT_NDOT": cov1_rtn[5][5]})

                object2.update({"X": pos_obj2[0]/1000, "Y": pos_obj2[1]/1000, "Z": pos_obj2[2]/1000,"X_DOT":vel_obj2[0]/1000,
                                "Y_DOT":vel_obj2[1]/1000,"Z_DOT":vel_obj2[2]/1000,"CR_R": cov2_rtn[0][0],"CT_R":cov2_rtn[1][0],
                                "CT_T":cov2_rtn[1][1],"CN_R":cov2_rtn[2][0],"CN_T":cov2_rtn[2][1],"CN_N":cov2_rtn[2][2],"CRDOT_R":cov2_rtn[3][0],
                                "CRDOT_T":cov2_rtn[3][1],"CRDOT_N":cov2_rtn[3][2],"CRDOT_RDOT":cov2_rtn[3][3],"CTDOT_R":cov2_rtn[4][0],
                                "CTDOT_T":cov2_rtn[4][1],"CTDOT_N":cov2_rtn[4][2],"CTDOT_RDOT":cov2_rtn[4][3],"CTDOT_TDOT":cov2_rtn[4][4],
                                "CNDOT_R":cov2_rtn[5][0],"CNDOT_T":cov2_rtn[5][1],"CNDOT_N":cov2_rtn[5][2],"CNDOT_RDOT":cov2_rtn[5][3],
                                "CNDOT_TDOT": cov2_rtn[5][4], "CNDOT_NDOT": cov2_rtn[5][5]})

                cdm_file = path.join(_output_path, f"""{object1["headers"]["OBJECT_ID"]}_{object2["headers"]["OBJECT_ID"]}_"""
                                     f"""{ca_utc[:19].replace("-", "").replace(":", "")}.cdm""")
                with open(cdm_file, "w") as fp:
                    fp.write(_cdm_template.render(obj1=object1, obj2=object2))

                event_min_dist, event_found = _critical_dist, False
                summary.append([object1["headers"]["EPHEMERIS_NAME"], object2["headers"]["EPHEMERIS_NAME"], cdm_file, ca_utc,
                                miss_dist, object1["RELATIVE_SPEED"]])

                if (_debug_mode):
                    with open(cdm_file.replace(".cdm", "_debug.json"), "w") as fp:
                        json.dump(_debug_data, fp, indent=2)
                    _debug_data = {}

    except Exception as exc:
        print(f"""Error {object1["headers"]["EPHEMERIS_NAME"]}, {object2["headers"]["EPHEMERIS_NAME"]}: {exc}""")
        return(None)

    return(summary)

def propagate(start_time, state1, state2, stop_time, step_size):
    cfg = [configure(prop_start=start_time,prop_initial_state=state1,prop_end=stop_time,prop_step=step_size,prop_inertial_frame=Frame.EME2000,
                     ocean_tides_degree=-1,ocean_tides_order=-1,solid_tides_sun=False,solid_tides_moon=False),
           configure(prop_start=start_time,prop_initial_state=state2,prop_end=stop_time,prop_step=step_size,prop_inertial_frame=Frame.EME2000,
                     ocean_tides_degree=-1,ocean_tides_order=-1,solid_tides_sun=False,solid_tides_moon=False)]

    return(propagate_orbits(cfg))

# Run apogee/perigee filter based on critical distance
def apogee_filter(s1, s2, delta):
    pos1, vel1 = s1[:3], s1[3:]
    pos2, vel2 = s2[:3], s2[3:]
    r1, v1sq = la.norm(pos1), np.dot(vel1, vel1)
    r2, v2sq = la.norm(pos2), np.dot(vel2, vel2)
    esc_vel = np.sqrt(2.0*_MU_EARTH/min(r1, r2))
    threshold = _critical_dist + esc_vel*delta

    a1 = _MU_EARTH/(2*_MU_EARTH/r1 - v1sq)
    a2 = _MU_EARTH/(2*_MU_EARTH/r2 - v2sq)
    h1v, h2v = np.cross(pos1, vel1), np.cross(pos2, vel2)
    h1sq, h2sq = np.dot(h1v, h1v), np.dot(h2v, h2v)
    e1sq = 1.0 - h1sq/(_MU_EARTH*a1)
    e2sq = 1.0 - h2sq/(_MU_EARTH*a2)
    e1 = np.sqrt(e1sq) if (e1sq > 0) else 0.0
    e2 = np.sqrt(e2sq) if (e2sq > 0) else 0.0

    rp1, ra1 = a1*(1.0 - e1), a1*(1.0 + e1)
    rp2, ra2 = a2*(1.0 - e2), a2*(1.0 + e2)
    if (abs(max(rp1, rp2) - min(ra1, ra2)) > threshold):
        return(False, rp1, rp2)
    else:
        return(True, rp1, rp2)

# Rodríguez, J. & Martínez, Francisco & Klinkrad, H.. “Collision Risk Assessment with a `Smart Sieve' Method”. (2002)
def sift(s1, s2, rp1, rp2, delta):
    rel_vec = [y - x for x, y in zip(s1, s2)]
    dist_vec, vel_vec = rel_vec[:3], rel_vec[3:]
    rel_dist, rel_vel = la.norm(dist_vec), la.norm(vel_vec)
    threshold = _critical_dist + np.sqrt(2*_MU_EARTH/min(la.norm(s1[:3]), la.norm(s2[:3])))*delta

    if (rel_dist > threshold):
        return(False, max(1, int((rel_dist - threshold)/(delta*np.sqrt(_MU_EARTH*(1/rp1 + 1/rp2))))))
    else:
        acc_threshold = _critical_dist + 9.80665*delta**2
        rminsq = rel_dist**2 - (np.dot(dist_vec, np.divide(vel_vec, rel_vel)))**2
        if (rminsq > acc_threshold**2):
            return(False, 1)
        else:
            acc_fine_threshold = acc_threshold + 0.5*abs(np.dot(dist_vec, np.divide(vel_vec, rel_vel)))*delta
            if (rel_dist > acc_fine_threshold):
                return(False, 1)
            else:
                return(True, 0)

# Function to find TCA using a binary search after the smart sieve process
def find_tca(time, state1, state2, delta, rp1, rp2):
    global _debug_mode, _debug_data

    distance = la.norm(np.array(state2)[:,:3] - np.array(state1)[:,:3], axis=1)
    min_idx = np.argmin(distance)
    min_dist = distance[min_idx]
    tca, state_ca1, state_ca2 = time[min_idx], state1[min_idx], state2[min_idx]

    if (min_idx == 2 and delta > 0.001):
        if (delta == time[1] - time[0]):
            inter_time, inter_state1, inter_state2 = time, state1, state2
        else:
            inter = interpolate_ephemeris(Frame.EME2000, time, state1, Frame.EME2000, time[0], time[-1], delta, interp_method=1)
            inter_time = [ix.time for ix in inter]
            inter_state1 = [ix.true_state for ix in inter]
            inter_state2 = [ix.true_state for ix in
                            interpolate_ephemeris(Frame.EME2000, time, state2, Frame.EME2000, time[0], time[-1], delta, interp_method=1)]

        for i in range(2, len(inter_time) - 2):
            sieve_pass, _ = sift(inter_state1[i], inter_state2[i], rp1, rp2, delta)

            if (sieve_pass):
                t = [inter_time[i + j] for j in range(-2, 3)]
                s1 = [inter_state1[i + j] for j in range(-2, 3)]
                s2 = [inter_state2[i + j] for j in range(-2, 3)]
                check = find_tca(t, s1, s2, delta/2.0, rp1, rp2)

                if (check and check[1] < min_dist):
                    tca, min_dist, state_ca1, state_ca2 = check

                    if (_debug_mode):
                        _debug_data = {"times": get_UTC_string(t, precision=6), "primaryStates": [list(s) for s in s1],
                                       "secondaryStates": [list(s) for s in s2], "tca": get_UTC_string(tca, precision=6),
                                       "primaryAtTca": list(state_ca1), "secondaryAtTca": list(state_ca2), "missDistance": min_dist}

        return(tca, min_dist, state_ca1, state_ca2)

    return(None)

def get_rotation(pos, vel):
    r_hat = pos/la.norm(pos)
    cross_pro = np.cross(pos, vel)
    n_hat = cross_pro/la.norm(cross_pro)
    t_hat = np.cross(n_hat, r_hat)
    rot3x3 = np.vstack((r_hat, t_hat, n_hat))

    rot6x6 = np.zeros((6, 6))
    rot6x6[:3,:3] = rot3x3
    rot6x6[3:,3:] = rot3x3

    return(rot3x3, rot6x6)

def get_covariance(obj, time, default):
    # If covariance in input file is not positive definite then return defaults
    try:
        index = bisect.bisect_right(obj["covTime"], time)
        if (index):
            cov = obj["cov"][index - 1]
            la.cholesky(ltr_to_matrix(cov))
            return(cov)
    except Exception as exc:
        print(f"""Warning {obj["headers"]["EPHEMERIS_NAME"]}: {exc}""")

    return(default)

# Chan's Pc approximation, outlined in Alfano 2013
def compute_pc(state1, state2, cov1, cov2):
    rel_state = np.subtract(state2, state1)
    Prel = np.add(cov1, cov2)[:3,:3]

    ux = np.divide(rel_state[:3], la.norm(rel_state[:3]))
    uy = np.cross(rel_state[:3], rel_state[3:])
    uy = np.divide(uy, la.norm(uy))

    T = np.vstack((ux, uy))
    w, v = la.eigh(T.dot(Prel).dot(T.transpose()))
    xbar, ybar = v.transpose().dot(T.dot(rel_state[:3]))
    sigx, sigy = w

    u1 = _body_radius**2/(np.sqrt(sigx)*np.sqrt(sigy))
    u2 = xbar**2/sigx + ybar**2/sigy
    pc = np.exp(-0.5*u2)*(1 - np.exp(-0.5*u1) + 0.5*u2*(1 - np.exp(-0.5*u1)*(1 + 0.5*u1)))

    return(0.0 if (np.isinf(pc)) else pc)
