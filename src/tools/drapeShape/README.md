# Drape vector geometry onto DGM1 and export DXF

`drape_shape.py` reads shapefile, DXF, or DWG geometry, finds the DGM1
GeoTIFF tiles that intersect its bounding box, extracts only those tiles from
a ZIP, samples terrain height at every geometry vertex, and writes 3D geometry
to DXF. Shapefiles use their `.prj` when present (otherwise EPSG:25832 is
assumed). DXF and DWG do not reliably carry a geographic CRS, so their
coordinates are automatically identified as UTM zone 32N or DHDN /
Gauss-Krüger zone 3 (EPSG:31467) from their coordinate ranges. Use
`--input-crs` to specify another CRS or override detection. CAD coordinates
are transformed to the DGM's ETRS89 / UTM zone 32N before tile selection and
elevation sampling.

DXF support includes points, lines, polylines, circles, arcs, ellipses and
splines in modelspace. Curved entities are sampled at 0.5 drawing-unit
flattening distance before draping. DWG files are converted to DXF using ODA
File Converter, which must be installed and available on `PATH` or provided
with `--dwg-converter`.

Tile selection uses names such as `dgm1_32_484_5554_1_he.tif`, interpreted as
the 1 km UTM square easting 484000–485000 m and northing 5554000–5555000 m.
The ZIP may contain subdirectories.

## Install and run

```powershell
py -m pip install -r requirements.txt
py drape_shape.py input.shp dgm_tiles.zip draped.dxf
py drape_shape.py input.dxf dgm_tiles.zip draped.dxf
py drape_shape.py input.dwg dgm_tiles.zip draped.dxf --dwg-converter /path/to/ODAFileConverter
```

For DXF/DWG data in another CRS or to override automatic detection, specify
its EPSG code. For example:

```powershell
py drape_shape.py input.dxf dgm_tiles.zip draped.dxf --input-crs EPSG:25832
```

Extracted tiles are kept in `draped_tiles` beside the DXF by default. Choose a
different directory with `--work-dir`. Points become DXF points; lines,
polylines, and polygon rings become 3D polylines.

All vertices must lie on valid DGM pixels. A missing tile, missing elevation,
or uncovered vertex stops the conversion with an error rather than writing an
incomplete DXF.
