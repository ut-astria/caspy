caspy (Conjunction Analysis and Screening with Python) is Python software
for ephemeris screening and conjunction analysis. 

Installation
------------

1. Install the orbdetpy <https://github.com/ut-astria/orbdetpy> `develop` branch.

2. Run `pip install -r requirements.txt` to install Python dependencies.

Usage
-----

1. caspy reads ephemerides from CCSDS OEM files and writes conjunction events
   to CDM files. Run `python caspy/caspy.py` for command line syntax.

2. Space-Track publishes operator ephmerides in the "Modified ITC" format detailed
   in <https://www.space-track.org/documents/Spaceflight_Safety_Handbook_for_Operators.pdf>.
   Use `caspy/meme_to_oem.py` to convert these files to OEM format.

Known Issues
------------

1. Probability of collision is work in progress and therefore COLLISION_PROBABILITY
   is not included in CDM files.

