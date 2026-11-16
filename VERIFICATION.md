# Verification — upgraded version

16 November 2026. `python test_project.py`: **8 tests passed**.

Checks cover dimension and upload rules, checkpoint/API score consistency, reconstruction of reported metrics, both binary output classes, explicit inconclusive behavior, graphics receiving no fabricated origin score, and disjoint balanced audit images. The fresh audit metrics use the fixed binary threshold on every image; uncertainty coverage is reported separately.

Browser tests verified the real and AI photo examples, content-type selection, the supplied screenshot's undetermined-origin result, and the final metrics dashboard. No original user AI-photo file was supplied, so its specific failure could not be reproduced from the screenshot alone. JavaScript syntax and Python compilation checks passed.

These checks verify implementation and internal experiment consistency, not universal detection performance.
