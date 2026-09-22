# Pinned RialTo object sources

This directory records the six real-object inputs used by the M1 object
catalog. `source_manifest.json` pins the RialToAssets repository revision,
download URL, SHA-256 digest, and repository-relative destination for every
input. The source files and generated USD files are deliberately kept out of
the runtime package until an asset-preparation command materializes them.

Runtime code is offline: it reads only resolved local files and never downloads
from GitHub. To explicitly fetch the pinned inputs, run:

```bash
PYTHONPATH=Go2Pvcnn python Go2Pvcnn/scripts/m1_rialto_object_assets.py \
  prepare --destination Go2Pvcnn/assets/m1_objects/rialto --allow-network
```

USDZ and GLB inputs require an installed USD/glTF converter. Conversion output
names are deterministic (`<source stem>.usd`) and every downloaded/generated
file should be hashed before it is promoted into a catalog.
