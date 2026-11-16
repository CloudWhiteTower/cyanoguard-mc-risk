# Attribution and licensing

The Python application, feature construction and temporal validation code are
SCAU-China's project code. This distribution adds packaging, input validation,
command-line entry points, synthetic examples and tests. Team code, documentation,
synthetic examples and the team-generated frozen model are provided under the
MIT license in `LICENSE`.

The model was fitted using water-quality observations from Ohio State University's
Stone Laboratory and OSUGI/Gibraltar Island Buoy. The original observations and
their row-level derivatives are not included. The MIT license does not relicense
those datasets. Obtain data from the original providers and follow their terms
and citation requirements; see [data sources](docs/data-sources.md).

Dependencies are installed separately and retain their own licenses:

| Dependency | Upstream license |
| --- | --- |
| XGBoost (`xgboost-cpu`) | Apache-2.0 |
| NumPy, pandas, SciPy, scikit-learn, joblib, threadpoolctl | BSD-3-Clause |
| PyYAML, netCDF4, cftime, six, openpyxl, et_xmlfile | MIT |
| python-dateutil | Apache-2.0 or BSD-3-Clause |
| certifi | MPL-2.0 |
| tzdata | Apache-2.0; bundled time-zone data has its own notices |

No upstream library source or binaries are vendored in this repository. Installed
wheels may carry further notices for their bundled native libraries. Their
license files remain part of those distributions. Pytest and the build tools
also retain their upstream licenses.
