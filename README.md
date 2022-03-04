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
