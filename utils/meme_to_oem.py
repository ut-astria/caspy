# meme_to_oem.py - Convert SpaceTrack Modified ITC to CCSDS OEM format.
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

from datetime import datetime
import glob
import multiprocessing
import numpy
from orbdetpy import configure, Frame
from orbdetpy.ccsds import export_OEM
from orbdetpy.conversion import get_J2000_epoch_offset, get_lvlh_rotation, ltr_to_matrix
from orbdetpy.rpc.messages_pb2 import EstimationOutput
from os import cpu_count, path
import sys

def convert(fname):
    try:
        toks = path.basename(fname).split("_")
        states, obj_id, obj_name = [], toks[1], toks[2]
        with open(fname, "r") as fp:
            lines = fp.read().splitlines()[4:]

        for idx in range(0, len(lines), 4):
            toks = lines[idx].split()
            epoch = get_J2000_epoch_offset(datetime.strptime(toks[0], "%Y%j%H%M%S.%f").strftime("%Y-%m-%dT%H:%M:%S.%f"))
            pv = [float(t)*1000.0 for t in toks[1:]]

            try:
                rot3 = get_lvlh_rotation(pv)
                rotation = numpy.zeros((6, 6))
                rotation[:3,:3] = rot3
                rotation[3:,3:] = rot3
                cov = numpy.array(ltr_to_matrix([float(t)*1E6 for t in " ".join(lines[idx + 1:idx + 4]).split()]))
                cov = rotation.transpose().dot(cov).dot(rotation)
                numpy.linalg.cholesky(cov)
                cov = [cov[i, j] for i in range(6) for j in range(i + 1)]
            except Exception as _:
                cov = []

            states.append(EstimationOutput(time=epoch, estimated_state=pv, propagated_covariance=cov))

        with open(f"""{fname.replace(".txt", ".oem")}""", "w") as fp:
            fp.write(export_OEM(configure(prop_inertial_frame=Frame.EME2000), states, obj_id, obj_name, add_prop_cov=True))
    except Exception as exc:
        print(f"Error {fname}: {exc}")

if (__name__ == "__main__"):
    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(processes=cpu_count()) as pool:
        inputs = glob.iglob(path.join(sys.argv[1] if (len(sys.argv) > 1) else ".", "*.txt"))
        for _ in pool.imap_unordered(convert, inputs):
            pass
