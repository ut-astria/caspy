# dedupe.py - Find duplicate OEM files and optionally delete stale ones.
# Copyright (C) 2022 University of Texas
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

from glob import iglob
from os import getenv, path, remove
import sys

def dedupe(root_dir, recursive=True, delete_stale_files=True):
    try:
        id_map = {}
        cat_file = getenv("CASPY_OBJECT_CATALOG", path.expanduser(path.join("~", "object_catalog.csv")))
        with open(cat_file, "r") as fp:
            for line in fp.read().splitlines():
                tok = line.split(",")
                id_map[tok[1]] = tok[0]
    except Exception as exc:
        print(f"Warning: {exc}")

    fresh_files, stale_files = {}, []

    for fname in (f for f in iglob(path.join(root_dir, "**"), recursive=recursive) if (path.isfile(f) and f.endswith(".oem"))):
        oid, start = None, None
        with open(fname, "r") as fp:
            for line in fp.read().splitlines():
                tok = [t.strip() for t in line.split("=")]
                if (tok[0] == "OBJECT_ID"):
                    oid = id_map.get(tok[1], tok[1])
                if (tok[0] == "START_TIME"):
                    start = tok[1]
                if (oid and start):
                    if (oid in fresh_files):
                        prev = fresh_files[oid]
                        if (prev[1] < start):
                            stale_files.append(prev[0])
                            fresh_files[oid] = (fname, start)
                        else:
                            stale_files.append(fname)
                    else:
                        fresh_files[oid] = (fname, start)
                    break

    if (delete_stale_files):
        for fname in stale_files:
            remove(fname)

    return(fresh_files, stale_files)

if (__name__ == "__main__"):
    root_dir = sys.argv[1] if (len(sys.argv) > 1) else "."
    dedupe(root_dir, recursive=True, delete_stale_files=True)
