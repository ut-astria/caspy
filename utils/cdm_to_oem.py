# cdm_to_oem.py - Extract object ephemeris from CDM files to OEM format.
# Copyright (C) 2023 University of Texas
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

import glob
import multiprocessing
import numpy
from orbdetpy import configure, Frame
from orbdetpy.ccsds import export_OEM
from orbdetpy.conversion import get_J2000_epoch_offset, get_lvlh_rotation, ltr_to_matrix
from orbdetpy.propagation import propagate_orbits
from orbdetpy.rpc.messages_pb2 import EstimationOutput
from os import cpu_count, path
import sys

def convert(fname):
    def state(obj):
        return((obj["X"], obj["Y"], obj["Z"], obj["X_DOT"], obj["Y_DOT"], obj["Z_DOT"]))

    def covariance(obj, state):
        rtn = (obj["CR_R"], obj["CT_R"], obj["CT_T"], obj["CN_R"], obj["CN_T"], obj["CN_N"], obj["CRDOT_R"], obj["CRDOT_T"],
               obj["CRDOT_N"], obj["CRDOT_RDOT"], obj["CTDOT_R"], obj["CTDOT_T"], obj["CTDOT_N"], obj["CTDOT_RDOT"], obj["CTDOT_TDOT"],
               obj["CNDOT_R"], obj["CNDOT_T"], obj["CNDOT_N"], obj["CNDOT_RDOT"], obj["CNDOT_TDOT"], obj["CNDOT_NDOT"])

        try:
            rot3x3 = get_lvlh_rotation(state)
            rot6x6 = numpy.zeros((6, 6))
            rot6x6[:3,:3] = rot3x3
            rot6x6[3:,3:] = rot3x3
            cov = rot6x6.transpose().dot(numpy.array(ltr_to_matrix(rtn))).dot(rot6x6)
            numpy.linalg.cholesky(cov)
            cov = [cov[i, j] for i in range(6) for j in range(i + 1)]
        except Exception as _:
            cov = []

        return(cov)

    def propagate(start, state1, state2, stop, step):
        cfg = [configure(prop_start=start,prop_initial_state=state1,prop_end=stop,prop_step=step,prop_inertial_frame=Frame.EME2000),
               configure(prop_start=start,prop_initial_state=state2,prop_end=stop,prop_step=step,prop_inertial_frame=Frame.EME2000)]

        return(cfg, propagate_orbits(cfg))

    try:
        cdm, key = {}, "COMMON"
        with open(fname, "r") as fp:
            for line in fp.read().splitlines():
                tok = [t.strip() for t in line.split("=")]

                if (tok[0] == "OBJECT" and tok[1] in ("OBJECT1", "OBJECT2")):
                    key = tok[1]
                    continue

                if ("[" in tok[1] and tok[1][-1] == "]"):
                    tok[1] = float(tok[1].split("[")[0])
                    if (tok[0] in ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")):
                        tok[1] *= 1000.0

                cdm.setdefault(key, {})[tok[0]] = tok[1]

        obj1, obj2 = cdm["OBJECT1"], cdm["OBJECT2"]
        epoch = get_J2000_epoch_offset(cdm["COMMON"]["TCA"])
        state1, state2 = state(obj1), state(obj2)

        _  , prop_rev = propagate(epoch, state1, state2, epoch - 300.0, -60.0)
        cfg, prop_fwd = propagate(epoch, state1, state2, epoch + 300.0, +60.0)

        output1, output2 = [], []
        for i in range(len(prop_rev[0].array) - 1, -1, -1):
            output1.append(EstimationOutput(time=prop_rev[0].array[i].time, estimated_state=prop_rev[0].array[i].true_state))
            output2.append(EstimationOutput(time=prop_rev[1].array[i].time, estimated_state=prop_rev[1].array[i].true_state))

        output1[-1].propagated_covariance[:] = covariance(obj1, state1)
        output2[-1].propagated_covariance[:] = covariance(obj2, state2)

        for i in range(1, len(prop_fwd[0].array)):
            output1.append(EstimationOutput(time=prop_fwd[0].array[i].time, estimated_state=prop_fwd[0].array[i].true_state))
            output2.append(EstimationOutput(time=prop_fwd[1].array[i].time, estimated_state=prop_fwd[1].array[i].true_state))

        out_dir, tca = path.dirname(fname), f"""{cdm["COMMON"]["TCA"][:19].replace("-", "").replace(":", "")}"""

        with open(path.join(out_dir, f"""TCA_{tca}_{obj1["OBJECT_DESIGNATOR"]}.oem"""), "w") as fp:
            fp.write(export_OEM(cfg[0], output1, obj1["OBJECT_DESIGNATOR"], obj1["OBJECT_NAME"], add_prop_cov=True))

        with open(path.join(out_dir, f"""TCA_{tca}_{obj2["OBJECT_DESIGNATOR"]}.oem"""), "w") as fp:
            fp.write(export_OEM(cfg[1], output2, obj2["OBJECT_DESIGNATOR"], obj2["OBJECT_NAME"], add_prop_cov=True))
    except Exception as exc:
        print(f"Error {fname}: {exc}")

if (__name__ == "__main__"):
    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(processes=cpu_count()) as pool:
        inputs = glob.iglob(path.join(sys.argv[1] if (len(sys.argv) > 1) else ".", "*.cdm"))
        for _ in pool.imap_unordered(convert, inputs):
            pass
