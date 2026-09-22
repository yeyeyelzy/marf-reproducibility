# Data and knowledge-source access

MIMIC-III and MIMIC-IV are controlled-access clinical databases.  This
repository does not redistribute raw records, derived patient data, code maps,
or any asset that can reveal patient-level information.  Each reproducer must
obtain their own authorized access through PhysioNet and comply with the
corresponding data-use agreement.

The CCS hierarchy is built from the public 2015 HCUP multi-level CCS mapping
for ICD-9-CM diagnoses.  MIMIC-IV preprocessing also requires an independently
obtained ICD-10-CM-to-ICD-9-CM GEM source.  These public knowledge files are
not committed here so that their license and version can be managed by the
user.

The supplied MIMIC-IV preprocessing entry point is a transparent starting
point for building private assets.  MIMIC-III release assets should be built
from the approved local preprocessing workflow and then checked against the
data contract in [DATA_FORMAT.md](DATA_FORMAT.md).  No private asset should be
copied into this repository, an issue, a pull request, or a public artifact.
