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
    global cdm_template
    cdm_template = Template(filename=path.join(path.dirname(path.realpath(__file__)), "template.cdm"))

def screen_pair(params):
    object1, object2, times, outputPath, interpOrder, interpTimeScale, criticalDistance, pos_sigma, vel_sigma, HBR = params
    states1, states2 = object1["states"], object2["states"]

    #Middle index of interpolation process
    indexOffset = int((interpOrder - 1)/2)

    #Apogee Perigee Filter
    underThreshold, rp1, rp2 = apogee_filter(states1[0], states2[0], times[1] - times[0], criticalDistance)
    if (not underThreshold):
        return(None)

    timeCloseApproach, eventOccurrence = 0, False
    i, eventMinDistance = indexOffset, criticalDistance
    interpTimeMin, interpS1Min, interpS2Min, summary = [], [], [], []

    #Iterate through each time step
    while (i < len(times) - indexOffset):
        #Run smart Sieve
        if (i >= min(len(states1), len(states2))):
            return(summary)
        passSieve, numIndexSkip = sift(states1[i], states2[i], rp1, rp2, times[1] - times[0], criticalDistance)

        #If pass continue. If fail, skip index
        if (not passSieve):
            i += numIndexSkip
            continue

        interpTime = [times[i + j] for j in range(-indexOffset, indexOffset + 1)]
        interpS1 = [states1[i + j] for j in range(-indexOffset, indexOffset + 1)]
        interpS2 = [states2[i + j] for j in range(-indexOffset, indexOffset + 1)]

        #If smart sieve passed, run binary search
        TCAResults = find_TCA(interpTime, interpS1, interpS2, times[1] - times[0], interpTimeScale, interpOrder, criticalDistance, rp1, rp2)

        # Event found
        if (TCAResults[1] < criticalDistance):
            #If minimum distance of event, save data
            if (TCAResults[1] < eventMinDistance):
                eventOccurrence = True
                interpTimeMin = interpTime
                interpS1Min = interpS1
                interpS2Min = interpS2
                eventMinDistance = TCAResults[1]
                timeCloseApproach= TCAResults[0]
        # Event not found
        else:
            # If event has just ended
            if (eventOccurrence):
                # Select closest state to propagate from
                if (timeCloseApproach > interpTimeMin[indexOffset]):
                    state1PreCloseApproach = interpS1Min[indexOffset]
                    state2PreCloseApproach = interpS2Min[indexOffset]
                    timePreCloseApproach = interpTimeMin[indexOffset]
                else:
                    state1PreCloseApproach = interpS1Min[indexOffset - 1]
                    state2PreCloseApproach = interpS2Min[indexOffset - 1]
                    timePreCloseApproach = interpTimeMin[indexOffset - 1]

                screen_start, screen_stop = get_UTC_string((times[0], times[-1]))
                time_pre_ca, time_ca = get_UTC_string((timePreCloseApproach, timeCloseApproach))
                message_id = f"""{time_ca}:{object1["objName"]}/{object2["objName"]}"""

                # Propagate states and covariances to TCA
                def_cov = [pos_sigma**2]*3 + [vel_sigma**2]*3
                cfg = [configure(prop_start=timePreCloseApproach, prop_initial_state=state1PreCloseApproach, prop_inertial_frame=Frame.EME2000,
                                 estm_filter=Filter.UNSCENTED_KALMAN, estm_covariance=find_covariance(object1["oemFile"], time_pre_ca, def_cov),
                                 drag_coefficient=Parameter(value=2.0, min=1.0, max=3.0, estimation=EstimationType.UNDEFINED),
                                 rp_coeff_reflection=Parameter(value=1.5, min=1.0, max=2.0, estimation=EstimationType.UNDEFINED),
                                 estm_process_noise=(1E-8,)*6, estm_DMC_corr_time=0, estm_DMC_sigma_pert=0),
                       configure(prop_start=timePreCloseApproach, prop_initial_state=state2PreCloseApproach, prop_inertial_frame=Frame.EME2000,
                                 estm_filter=Filter.UNSCENTED_KALMAN, estm_covariance=find_covariance(object2["oemFile"], time_pre_ca, def_cov),
                                 drag_coefficient=Parameter(value=2.0, min=1.0, max=3.0, estimation=EstimationType.UNDEFINED),
                                 rp_coeff_reflection=Parameter(value=1.5, min=1.0, max=2.0, estimation=EstimationType.UNDEFINED),
                                 estm_process_noise=(1E-8,)*6, estm_DMC_corr_time=0, estm_DMC_sigma_pert=0)]
                cfg[0].measurements[MeasurementType.POSITION].error[:] = [100.0]*3
                cfg[1].measurements[MeasurementType.POSITION].error[:] = [100.0]*3

                fit = determine_orbit(cfg, [[build_measurement(timeCloseApproach, "", [])]]*2)
                state_ca1, state_ca2 = fit[0][-1].estimated_state, fit[1][-1].estimated_state
                prop_cov1 = np.array(ltr_to_matrix(fit[0][-1].propagated_covariance))
                prop_cov2 = np.array(ltr_to_matrix(fit[1][-1].propagated_covariance))
                pos_obj1, vel_obj1 = state_ca1[:3], state_ca1[3:]
                pos_obj2, vel_obj2 = state_ca2[:3], state_ca2[3:]

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
                relative_pos = rotation.dot(np.subtract(pos_obj2, pos_obj1))
                relative_vel = rotation.dot(np.subtract(vel_obj2, vel_obj1))
                coll_prob = compute_Pc(state_ca1, state_ca2, prop_cov1, prop_cov2, HBR)

                obj1_map = {"CREATION_DATE":datetime.now(timezone.utc).isoformat(timespec="milliseconds")[:-6],"MESSAGE_ID":message_id,"TCA":time_ca,
                            "MISS_DISTANCE":la.norm(np.subtract(pos_obj1,pos_obj2)),"RELATIVE_SPEED":la.norm(np.subtract(vel_obj1,vel_obj2)),
                            "RELATIVE_POSITION_R":relative_pos[0],"RELATIVE_POSITION_T":relative_pos[1],"RELATIVE_POSITION_N":relative_pos[2],
                            "RELATIVE_VELOCITY_R":relative_vel[0],"RELATIVE_VELOCITY_T":relative_vel[1],"RELATIVE_VELOCITY_N":relative_vel[2],
                            "START_SCREEN_PERIOD": screen_start, "STOP_SCREEN_PERIOD": screen_stop, "COLLISION_PROBABILITY": coll_prob,
                            "OBJECT_DESIGNATOR": object1["objID"],"OBJECT_NAME": object1["objName"],"TIME_LASTOB_START":object1["startTime"],
                            "TIME_LASTOB_END":object1["endTime"],"X":pos_obj1[0]/1000,"Y":pos_obj1[1]/1000,"Z":pos_obj1[2]/1000,
                            "X_DOT":vel_obj1[0]/1000,"Y_DOT":vel_obj1[1]/1000,"Z_DOT":vel_obj1[2]/1000,"CR_R":cov1_rtn[0][0],"CT_R":cov1_rtn[1][0],
                            "CT_T":cov1_rtn[1][1],"CN_R":cov1_rtn[2][0],"CN_T":cov1_rtn[2][1],"CN_N":cov1_rtn[2][2],"CRDOT_R":cov1_rtn[3][0],
                            "CRDOT_T":cov1_rtn[3][1],"CRDOT_N":cov1_rtn[3][2],"CRDOT_RDOT":cov1_rtn[3][3],"CTDOT_R":cov1_rtn[4][0],
                            "CTDOT_T":cov1_rtn[4][1],"CTDOT_N":cov1_rtn[4][2],"CTDOT_RDOT":cov1_rtn[4][3],"CTDOT_TDOT":cov1_rtn[4][4],
                            "CNDOT_R":cov1_rtn[5][0],"CNDOT_T":cov1_rtn[5][1],"CNDOT_N":cov1_rtn[5][2],"CNDOT_RDOT":cov1_rtn[5][3],
                            "CNDOT_TDOT":cov1_rtn[5][4],"CNDOT_NDOT":cov1_rtn[5][5]}

                obj2_map = {"OBJECT_DESIGNATOR": object2["objID"], "OBJECT_NAME": object2["objName"], "TIME_LASTOB_START": object2["startTime"],
                            "TIME_LASTOB_END":object2["endTime"],"X":pos_obj2[0]/1000,"Y":pos_obj2[1]/1000,"Z":pos_obj2[2]/1000,
                            "X_DOT":vel_obj2[0]/1000,"Y_DOT":vel_obj2[1]/1000,"Z_DOT":vel_obj2[2]/1000,"CR_R": cov2_rtn[0][0],"CT_R":cov2_rtn[1][0],
                            "CT_T":cov2_rtn[1][1],"CN_R":cov2_rtn[2][0],"CN_T":cov2_rtn[2][1],"CN_N":cov2_rtn[2][2],"CRDOT_R":cov2_rtn[3][0],
                            "CRDOT_T":cov2_rtn[3][1],"CRDOT_N":cov2_rtn[3][2],"CRDOT_RDOT":cov2_rtn[3][3],"CTDOT_R":cov2_rtn[4][0],
                            "CTDOT_T":cov2_rtn[4][1],"CTDOT_N":cov2_rtn[4][2],"CTDOT_RDOT":cov2_rtn[4][3],"CTDOT_TDOT":cov2_rtn[4][4],
                            "CNDOT_R":cov2_rtn[5][0],"CNDOT_T":cov2_rtn[5][1],"CNDOT_N":cov2_rtn[5][2],"CNDOT_RDOT":cov2_rtn[5][3],
                            "CNDOT_TDOT":cov2_rtn[5][4],"CNDOT_NDOT":cov2_rtn[5][5]}

                eventMinDistance, eventOccurrence = criticalDistance, False
                cdm_file = path.join(outputPath, f"""{object1["objName"]}_{object2["objName"]}_{time_ca[:19].replace("-", "").replace(":", "")}.cdm""")
                with open(cdm_file, "w") as fp:
                    fp.write(cdm_template.render(obj1=obj1_map, obj2=obj2_map))
                summary.append([object1["oemFile"], object2["oemFile"], cdm_file, time_ca, obj1_map["MISS_DISTANCE"], obj1_map["RELATIVE_SPEED"]])
        i+=1
    return(summary)

# Function to read in object states, and run apogee/perigee filter based upon critical distance
def apogee_filter(s1, s2, delt, criticalDistance):
    pos1, vel1 = s1[:3], s1[3:]
    pos2, vel2 = s2[:3], s2[3:]
    r1, v1sq = la.norm(pos1), np.dot(vel1, vel1)
    r2, v2sq = la.norm(pos2), np.dot(vel2, vel2)
    vEscape = np.sqrt(2*_MU_EARTH/min(r1, r2))
    thresholdDistance = criticalDistance + vEscape*delt

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
def sift(s1, s2, rp1, rp2, delt, criticalDistance):
    relVec = [y - x for x, y in zip(s1, s2)]
    relDistanceVec, relVelocityVec = relVec[:3], relVec[3:]
    relDistance, relVelocity = la.norm(relDistanceVec), la.norm(relVelocityVec)
    vEscape = np.sqrt(2*_MU_EARTH/min(la.norm(s1[:3]), la.norm(s2[:3])))
    thresholdDistance = criticalDistance + vEscape*delt
    if (relDistance > thresholdDistance):
        return(False, max(1, int((relDistance - thresholdDistance)/(delt*np.sqrt(_MU_EARTH*(1/rp1 + 1/rp2))))))
    else:
        accThresholdDistance = criticalDistance + 9.80665*delt**2
        rminsq = relDistance**2 - (np.dot(relDistanceVec, np.divide(relVelocityVec, relVelocity)))**2
        if (rminsq > accThresholdDistance**2):
            return(False, 1)
        else:
            accFineThresholdDistance = accThresholdDistance + 0.5*abs(np.dot(relDistanceVec, np.divide(relVelocityVec, relVelocity)))*delt
            if (relDistance > accFineThresholdDistance):
                return(False, 1)
            else:
                return(True, 0)

# Function to find TCA for two OEM files after the Smart Sieve Process
# Binary searach algorithm used, which utilizes orbdetpy interpolate_ephemeris function
def find_TCA(t, s1, s2, delt, interpTimeScale, interpOrder, criticalDistance, rp1, rp2):
    indexOffset = int((interpOrder - 1)/2)
    distance = la.norm(np.array(s2)[:,:3] - np.array(s1)[:,:3], axis=1)
    minDistance = min(distance)
    minIndex = np.where(distance == minDistance)[0][0]
    timeCloseApproach = t[minIndex]
    if (minIndex == indexOffset and delt > interpTimeScale):
        interp1 = interpolate_ephemeris(Frame.EME2000, t, s1, interpOrder, Frame.EME2000, t[0], t[-1], delt)
        interp2 = interpolate_ephemeris(Frame.EME2000, t, s2, interpOrder, Frame.EME2000, t[0], t[-1], delt)
        for i in range(indexOffset, len(interp1) - indexOffset):
            passSieve, numIndexSkip = sift(interp1[i].true_state, interp2[i].true_state, rp1, rp2, delt, criticalDistance)
            if (passSieve):
                interpTime = [interp1[i + j].time for j in range(-indexOffset, indexOffset + 1)]
                interpS1 = [interp1[i + j].true_state for j in range(-indexOffset, indexOffset + 1)]
                interpS2 = [interp2[i + j].true_state for j in range(-indexOffset, indexOffset + 1)]
                checkTCA = find_TCA(interpTime, interpS1, interpS2, delt/2, interpTimeScale, interpOrder, criticalDistance, rp1, rp2)
                if (checkTCA[1] < minDistance):
                    timeCloseApproach, minDistance = checkTCA
    return(timeCloseApproach, minDistance)

def find_covariance(oem_file, time, default):
    with open(oem_file, "r") as fp:
        lines = [l.strip() for l in fp.readlines() if (l.strip())]

    start, ep0_idx = False, -1
    for idx, line in enumerate(lines):
        if (line.startswith("COVARIANCE_STOP")):
            break
        if (start and line.startswith("EPOCH")):
            epoch1 = line.split("=")[-1].strip()
            if (ep0_idx == -1 or time >= epoch1):
                ep0_idx = idx
            if (time < epoch1):
                return([float(t)*1E6 for t in " ".join((l for l in lines[ep0_idx + 1:idx] if not ("=" in l or l.startswith("COMMENT")))).split()])
        if (line.startswith("COVARIANCE_START")):
            start = True
    return(default)

# Produces CDM with input arguments
# Chan's Approximation, outlined in Alfano 2013
def compute_Pc(state1, state2, covar1, covar2, HBR):
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

    u1 = HBR**2/(sigx*sigy)
    u2 = xbar**2/sigx**2 + ybar**2/sigy**2
    term0 = np.exp(-u2/2)*(1 - np.exp(-u1/2))
    term1 = u1**2*u2*np.exp(0.25*u2*(u1 - 2))/8
    return(term0 + term1)
