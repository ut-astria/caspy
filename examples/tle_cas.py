# tle_cas.py - Screen TLEs for close approaches.
# Copyright (C) 2022-2023 University of Texas
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

import caspy
from datetime import datetime
import os

if (__name__ == "__main__"):
    # Primary object = GLAST
    primary = [["0 GLAST",
                "1 33053U 08029A   22059.73845930  .00001259  00000-0  54777-4 0  9991",
                "2 33053  25.5816 188.2914 0012165   2.5099 357.5496 15.11835419757226"]]

    # Secondary objects = some Starlinks
    secondary = [["0 STARLINK-3053",
                  "1 49149U 21082V   22059.91667824  .00029526  00000-0  24719-2 0  9990",
                  "2 49149  70.0005  40.2661 0003437 284.2373 109.9727 14.98329457  3578"],
                 ["0 STARLINK-3304",
                  "1 50161U 21125F   22059.91667824 -.00662428  00000-0 -41764-1 0  9998",
                  "2 50161  53.2176  17.1960 0001915  34.6405  24.8025 15.11099020  3632"],
                 ["0 STARLINK-2649",
                  "1 48680U 21044AU  22059.74267740 -.00000139  00000-0  95435-5 0  9997",
                  "2 48680  53.0554 230.4645 0001634  69.2317 290.8847 15.06401102 42291"]]

    start_time = datetime.utcnow()

    # Screen for approaches < 50 km over a 24 hr time window
    # CDM files will be written to your home directory
    caspy.run_tle_cas(primary, secondary, os.path.expanduser("~"), distance=50.0E3, window=24.0, cache_data=True)

    tmin, tsec = divmod((datetime.utcnow() - start_time).total_seconds(), 60.0)
    print(f"Elapsed time = {tmin:.0f} min {tsec:.1f} sec")
