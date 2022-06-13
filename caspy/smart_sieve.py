# smart_sieve.py - Conjunction analysis using the smart sieve algorithm.
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

import bisect
from datetime import datetime, timezone
import json
from mako.template import Template
import numpy as np
import numpy.linalg as la
from orbdetpy import configure, EstimationType, Frame, Filter, build_measurement, MeasurementType
from orbdetpy.conversion import get_J2000_epoch_offset, get_UTC_string, ltr_to_matrix
from orbdetpy.estimation import determine_orbit
from orbdetpy.utilities import interpolate_ephemeris
from orbdetpy.rpc.messages_pb2 import Parameter
from os import path

_MU_EARTH = 3.986004418E14

def init_process():
    global _cdm_template
    _cdm_template = Template(filename=path.join(path.dirname(path.realpath(__file__)), "template.cdm"))

def screen_pair(params):
    try:
        object1, object2, times, outputPath, interpOrder, interpTimeScale, critical_dist, pos_sigma, vel_sigma, body_radius = params
        states1, states2 = object1["states"], object2["states"]

        # Middle index of interpolation process
        indexOffset = int((interpOrder - 1)/2)

        # Apogee Perigee Filter
        underThreshold, rp1, rp2 = apogee_filter(states1[0], states2[0], times[1] - times[0], critical_dist)
        if (not underThreshold):
            return(None)

        time_ca, event_found = 0, False
        interpTimeMin, interpS1Min, interpS2Min, summary = [], [], [], []
        index, event_min_dist, max_index = indexOffset, critical_dist, len(times) - indexOffset

        # Iterate through each time step
        while (index < max_index):
            # Run smart Sieve
            passSieve, numIndexSkip = sift(states1[index], states2[index], rp1, rp2, times[1] - times[0], critical_dist)

            # If pass continue. If fail, skip index
            if (not passSieve):
                index += numIndexSkip
                continue

            interpTime = [times[index + j] for j in range(-indexOffset, indexOffset + 1)]
            interpS1 = [states1[index + j] for j in range(-indexOffset, indexOffset + 1)]
            interpS2 = [states2[index + j] for j in range(-indexOffset, indexOffset + 1)]

            # If smart sieve passed, run binary search
            tca_result = find_TCA(interpTime, interpS1, interpS2, times[1] - times[0], interpTimeScale, interpOrder, critical_dist, rp1, rp2)

            # Event found
            if (tca_result[1] < event_min_dist):
                event_found = True
                interpTimeMin = interpTime
                interpS1Min = interpS1
                interpS2Min = interpS2
                time_ca, event_min_dist = tca_result

            # If event has just ended
            if (event_found and (tca_result[1] >= event_min_dist or index == max_index - 1)):
                # Select closest state to propagate from
                if (time_ca > interpTimeMin[indexOffset]):
                    state1_pre_ca = interpS1Min[indexOffset]
                    state2_pre_ca = interpS2Min[indexOffset]
                    time_pre_ca = interpTimeMin[indexOffset]
                else:
                    state1_pre_ca = interpS1Min[indexOffset - 1]
                    state2_pre_ca = interpS2Min[indexOffset - 1]
                    time_pre_ca = interpTimeMin[indexOffset - 1]

                ca_utc = get_UTC_string(time_ca)
                screen_start, screen_stop = get_UTC_string((times[0], times[-1]))

                # Propagate states and covariances to TCA
                def_cov = [pos_sigma**2]*3 + [vel_sigma**2]*3
                cfg = [configure(prop_start=time_pre_ca, prop_initial_state=state1_pre_ca, prop_inertial_frame=Frame.EME2000,
                                 ocean_tides_degree=-1, ocean_tides_order=-1, solid_tides_sun=False, solid_tides_moon=False,
                                 estm_filter=Filter.UNSCENTED_KALMAN, estm_covariance=get_covariance(object1, time_pre_ca, def_cov),
                                 drag_coefficient=Parameter(value=2.0, min=1.0, max=3.0, estimation=EstimationType.UNDEFINED),
                                 rp_coeff_reflection=Parameter(value=1.5, min=1.0, max=2.0, estimation=EstimationType.UNDEFINED),
                                 estm_process_noise=(1E-8,)*6, estm_DMC_corr_time=0, estm_DMC_sigma_pert=0),
                       configure(prop_start=time_pre_ca, prop_initial_state=state2_pre_ca, prop_inertial_frame=Frame.EME2000,
                                 ocean_tides_degree=-1, ocean_tides_order=-1, solid_tides_sun=False, solid_tides_moon=False,
                                 estm_filter=Filter.UNSCENTED_KALMAN, estm_covariance=get_covariance(object2, time_pre_ca, def_cov),
                                 drag_coefficient=Parameter(value=2.0, min=1.0, max=3.0, estimation=EstimationType.UNDEFINED),
                                 rp_coeff_reflection=Parameter(value=1.5, min=1.0, max=2.0, estimation=EstimationType.UNDEFINED),
                                 estm_process_noise=(1E-8,)*6, estm_DMC_corr_time=0, estm_DMC_sigma_pert=0)]
                cfg[0].measurements[MeasurementType.POSITION].error[:] = [100.0]*3
                cfg[1].measurements[MeasurementType.POSITION].error[:] = [100.0]*3

                fit = determine_orbit(cfg, [[build_measurement(time_ca, "", [])]]*2)
                state_ca1, state_ca2 = fit[0][-1].estimated_state, fit[1][-1].estimated_state
                prop_cov1 = np.array(ltr_to_matrix(fit[0][-1].propagated_covariance))
                prop_cov2 = np.array(ltr_to_matrix(fit[1][-1].propagated_covariance))
                pos_obj1, vel_obj1 = state_ca1[:3], state_ca1[3:]
                pos_obj2, vel_obj2 = state_ca2[:3], state_ca2[3:]
                miss_dist = la.norm(np.subtract(pos_obj1, pos_obj2))

                r_hat = pos_obj1/la.norm(pos_obj1)
                cross_pro = np.cross(pos_obj1, vel_obj1)
                n_hat = cross_pro/la.norm(cross_pro)
                t_hat = np.cross(n_hat, r_hat)
                rotation = np.vstack((r_hat, t_hat, n_hat))

                R = np.zeros((6, 6))
                R[:3,:3] = rotation
                R[3:,3:] = rotation
                cov1_rtn = R.dot(prop_cov1).dot(R.transpose())
                cov2_rtn = R.dot(prop_cov2).dot(R.transpose())
                rel_pos = rotation.dot(np.subtract(pos_obj2, pos_obj1))
                rel_vel = rotation.dot(np.subtract(vel_obj2, vel_obj1))
                coll_prob = compute_Pc(state_ca1, state_ca2, prop_cov1, prop_cov2, body_radius)

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

                index += 2*indexOffset
                event_min_dist, event_found = critical_dist, False
                if (miss_dist < critical_dist):
                    cdm_file = path.join(outputPath, f"""{object1["headers"]["OBJECT_ID"]}_{object2["headers"]["OBJECT_ID"]}_"""
                                         f"""{ca_utc[:19].replace("-", "").replace(":", "")}.cdm""")
                    with open(cdm_file, "w") as fp:
                        fp.write(_cdm_template.render(obj1=object1, obj2=object2))
                    summary.append([object1["headers"]["EPHEMERIS_NAME"], object2["headers"]["EPHEMERIS_NAME"], cdm_file, ca_utc,
                                    miss_dist, object1["RELATIVE_SPEED"]])
            else:
                index += 1
    except Exception as exc:
        print(f"""{object1["headers"]["EPHEMERIS_NAME"]}, {object2["headers"]["EPHEMERIS_NAME"]}: {exc}""")
        return(None)
    return(summary)

# Function to read in object states, and run apogee/perigee filter based upon critical distance
def apogee_filter(s1, s2, delta, critical_dist):
    pos1, vel1 = s1[:3], s1[3:]
    pos2, vel2 = s2[:3], s2[3:]
    r1, v1sq = la.norm(pos1), np.dot(vel1, vel1)
    r2, v2sq = la.norm(pos2), np.dot(vel2, vel2)
    vEscape = np.sqrt(2*_MU_EARTH/min(r1, r2))
    thresholdDistance = critical_dist + vEscape*delta

    a1 = _MU_EARTH/(2*_MU_EARTH/r1 - v1sq)
    a2 = _MU_EARTH/(2*_MU_EARTH/r2 - v2sq)
    h1v, h2v = np.cross(pos1, vel1), np.cross(pos2, vel2)
    h1sq, h2sq = np.dot(h1v, h1v), np.dot(h2v, h2v)
    e1sq = 1 - h1sq/(_MU_EARTH*a1)
    e2sq = 1 - h2sq/(_MU_EARTH*a2)
    e1 = np.sqrt(e1sq) if (e1sq > 0) else 0
    e2 = np.sqrt(e2sq) if (e2sq > 0) else 0

    rp1, ra1 = a1*(1 - e1), a1*(1 + e1)
    rp2, ra2 = a2*(1 - e2), a2*(1 + e2)
    if (abs(max(rp1, rp2) - min(ra1, ra2)) > thresholdDistance):
        return(False, rp1, rp2)
    else:
        return(True, rp1, rp2)

# Rodríguez, J. & Martínez, Francisco & Klinkrad, H.. “Collision Risk Assessment with a `Smart Sieve' Method”. (2002)
def sift(s1, s2, rp1, rp2, delta, critical_dist):
    relVec = [y - x for x, y in zip(s1, s2)]
    relDistanceVec, relVelocityVec = relVec[:3], relVec[3:]
    relDistance, relVelocity = la.norm(relDistanceVec), la.norm(relVelocityVec)
    vEscape = np.sqrt(2*_MU_EARTH/min(la.norm(s1[:3]), la.norm(s2[:3])))
    thresholdDistance = critical_dist + vEscape*delta
    if (relDistance > thresholdDistance):
        return(False, max(1, int((relDistance - thresholdDistance)/(delta*np.sqrt(_MU_EARTH*(1/rp1 + 1/rp2))))))
    else:
        accThresholdDistance = critical_dist + 9.80665*delta**2
        rminsq = relDistance**2 - (np.dot(relDistanceVec, np.divide(relVelocityVec, relVelocity)))**2
        if (rminsq > accThresholdDistance**2):
            return(False, 1)
        else:
            accFineThresholdDistance = accThresholdDistance + 0.5*abs(np.dot(relDistanceVec, np.divide(relVelocityVec, relVelocity)))*delta
            if (relDistance > accFineThresholdDistance):
                return(False, 1)
            else:
                return(True, 0)

# Function to find TCA for two OEM files after the Smart Sieve Process
# Binary searach algorithm used, which utilizes orbdetpy interpolate_ephemeris function
def find_TCA(t, s1, s2, delta, interpTimeScale, interpOrder, critical_dist, rp1, rp2):
    indexOffset = int((interpOrder - 1)/2)
    distance = la.norm(np.array(s2)[:,:3] - np.array(s1)[:,:3], axis=1)
    minDistance = min(distance)
    minIndex = np.where(distance == minDistance)[0][0]
    timeCloseApproach = t[minIndex]
    if (minIndex == indexOffset and delta > interpTimeScale):
        interp1 = interpolate_ephemeris(Frame.EME2000, t, s1, interpOrder, Frame.EME2000, t[0], t[-1], delta)
        interp2 = interpolate_ephemeris(Frame.EME2000, t, s2, interpOrder, Frame.EME2000, t[0], t[-1], delta)
        for i in range(indexOffset, len(interp1) - indexOffset):
            passSieve, numIndexSkip = sift(interp1[i].true_state, interp2[i].true_state, rp1, rp2, delta, critical_dist)
            if (passSieve):
                interpTime = [interp1[i + j].time for j in range(-indexOffset, indexOffset + 1)]
                interpS1 = [interp1[i + j].true_state for j in range(-indexOffset, indexOffset + 1)]
                interpS2 = [interp2[i + j].true_state for j in range(-indexOffset, indexOffset + 1)]
                checkTCA = find_TCA(interpTime, interpS1, interpS2, delta/2, interpTimeScale, interpOrder, critical_dist, rp1, rp2)
                if (checkTCA[1] < minDistance):
                    timeCloseApproach, minDistance = checkTCA
    return(timeCloseApproach, minDistance)

def get_covariance(obj, time, default):
    index = bisect.bisect_right(obj["covTime"], time)
    return(obj["cov"][index - 1] if (index) else default)

# Produces CDM with input arguments
# Chan's Approximation, outlined in Alfano 2013
def compute_Pc(state1, state2, covar1, covar2, body_radius):
    stateRel = np.subtract(state2, state1)
    subindex = np.ix_([0,1,2], [0,1,2])
    Prel = np.add(covar1[subindex], covar2[subindex])

    ux = np.divide(stateRel[:3], la.norm(stateRel[:3]))
    uy = np.cross(stateRel[:3], stateRel[3:])
    uy = np.divide(uy, la.norm(uy))

    T = np.vstack((ux, uy))
    w, v = la.eigh(T.dot(Prel).dot(T.transpose()))
    stateRot = v.transpose().dot(T.dot(stateRel[:3]))
    xbar, ybar = stateRot
    sigx, sigy = np.sqrt(w)

    u1 = body_radius**2/(sigx*sigy)
    u2 = xbar**2/sigx**2 + ybar**2/sigy**2
    pc = np.exp(-u2/2)*(1 - np.exp(-u1/2)) + u1**2*u2*np.exp(0.25*u2*(u1 - 2))/8
    return(0.0 if (np.isinf(pc)) else pc)
