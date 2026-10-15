"""Geo-inference: Extract features from high-resolution geospatial imagery using foundation models"""

import importlib.util
import os

__author__ = "Victor Alhassan"
__version__ = "3.2.4"

# Use the PROJ/GDAL data bundled with the rasterio wheel, overriding any PROJ_DATA/GDAL_DATA
# leaked from an outer conda env (a mismatched proj.db breaks CRS lookups).
_rasterio_spec = importlib.util.find_spec("rasterio")
if _rasterio_spec is not None and _rasterio_spec.origin:
    _rasterio_dir = os.path.dirname(_rasterio_spec.origin)
    for _var, _sub in (("PROJ_DATA", "proj_data"), ("GDAL_DATA", "gdal_data")):
        _path = os.path.join(_rasterio_dir, _sub)
        if os.path.isdir(_path):
            os.environ[_var] = _path
