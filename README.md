**caspy** (Conjunction Analysis and Screening with Python) is Python software
for ephemeris screening and conjunction analysis. **caspy** reads ephemerides
from CCSDS OEM files and writes conjunction events to CDM files. 

Installation
------------

1. Install the orbdetpy <https://github.com/ut-astria/orbdetpy> `develop` branch.

2. Run `pip install -r requirements.txt` to install Python dependencies.

Usage
-----

1. Run `python caspy/cas.py` for command line syntax. You can also import the `caspy`
   package and call the `run_cas()` function from your own Python code. The `run_cas()`
   arguments mirror the CLI parameters.

2. Space-Track publishes operator ephmerides in the "Modified ITC" format detailed
   in <https://www.space-track.org/documents/Spaceflight_Safety_Handbook_for_Operators.pdf>.
   Use `utils/meme_to_oem.py` to convert these files to OEM format.

3. Download the UT object catalog ID file to ensure caspy uses object identifiers consistently. The file must be updated frequently to account for launches etc. Provide the fully qualified file name in the environment variable `CASPY_OBJECT_CATALOG` before running caspy. For example:

```
export CASPY_OBJECT_CATALOG=$HOME/object_catalog.csv

curl -o $CASPY_OBJECT_CATALOG http://astria.tacc.utexas.edu/AstriaGraph/OD_stats/object_catalog.csv
```

Docker 
______

You will first need to clone [orbdetpy](https://github.com/ut-astria/orbdetpy) and build its docker image as posted [here](https://github.com/ut-astria/orbdetpy#docker). 

```bash
docker build -t caspy:core .
docker run --rm caspy:core -h
```

Examples
--------

1. `examples/tle_cas.py` demonstrates TLE screening functionality.
