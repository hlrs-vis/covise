#!/usr/bin/env python3
"""Drape shapefile vertices over German DGM1 GeoTIFF tiles and export DXF."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import ezdxf
import numpy as np
import rasterio
import shapefile
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge

TILE_RE = re.compile(r"dgm1_32_(\d+)_(\d+)_1_he\.tif$", re.IGNORECASE)
UTM32N = CRS.from_epsg(25832)


def tile_bounds(member: str) -> tuple[float, float, float, float] | None:
    """Return (west, south, east, north) for the 1 km tile encoded in its name."""
    match = TILE_RE.search(PurePosixPath(member).name)
    if not match:
        return None
    east_km, north_km = map(int, match.groups())
    west, south = east_km * 1000.0, north_km * 1000.0
    return west, south, west + 1000.0, south + 1000.0


def intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def read_shape_bounds(path: Path, target_crs: CRS, cad_crs: CRS | None = None) -> tuple[float, float, float, float]:
    if path.suffix.lower() != ".shp":
        coords = [pt for _, points, _ in iter_cad_features(path) for pt in points]
        if not coords:
            raise ValueError(f"No supported geometry found in {path.name}")
        if cad_crs and not cad_crs.equals(target_crs):
            transformer = Transformer.from_crs(cad_crs, target_crs, always_xy=True)
            coords = [transformer.transform(x, y) for x, y in coords]
        xmin = min(p[0] for p in coords); ymin = min(p[1] for p in coords)
        xmax = max(p[0] for p in coords); ymax = max(p[1] for p in coords)
        return xmin, ymin, xmax, ymax
    reader = shapefile.Reader(str(path))
    try:
        if not reader.shapes():
            raise ValueError("Input shapefile contains no shapes")
        xmin, ymin, xmax, ymax = reader.bbox
    finally:
        reader.close()
    prj = path.with_suffix(".prj")
    if not prj.exists():
        # The command's documented input contract is UTM zone 32N.
        return xmin, ymin, xmax, ymax
    source_crs = CRS.from_wkt(prj.read_text(encoding="utf-8", errors="replace"))
    if source_crs != target_crs:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        # Transform all bbox corners; densification covers non-linear projections.
        xs, ys = [], []
        for i in range(21):
            t = i / 20
            for x, y in ((xmin + t * (xmax - xmin), ymin),
                         (xmin + t * (xmax - xmin), ymax),
                         (xmin, ymin + t * (ymax - ymin)),
                         (xmax, ymin + t * (ymax - ymin))):
                tx, ty = transformer.transform(x, y)
                xs.append(tx); ys.append(ty)
        return min(xs), min(ys), max(xs), max(ys)
    return xmin, ymin, xmax, ymax


def iter_cad_features(path: Path):
    """Yield supported DXF entities as (kind, XY vertices, closed)."""
    try:
        doc = ezdxf.readfile(path)
    except ezdxf.DXFError as exc:
        raise ValueError(f"Cannot read DXF {path}: {exc}") from exc
    from ezdxf.path import make_path

    supported = {"POINT", "LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "ELLIPSE", "SPLINE"}
    for entity in doc.modelspace():
        kind = entity.dxftype()
        if kind not in supported:
            continue
        if kind == "POINT":
            p = entity.dxf.location
            yield "point", [(p.x, p.y)], False
            continue
        if kind == "LINE":
            a, b = entity.dxf.start, entity.dxf.end
            yield "line", [(a.x, a.y), (b.x, b.y)], False
            continue
        try:
            path_obj = make_path(entity)
            points = [(p.x, p.y) for p in path_obj.flattening(0.5)]
        except (TypeError, ValueError, AttributeError):
            continue
        closed_value = getattr(entity, "closed", getattr(entity, "is_closed", False))
        closed = bool(closed_value() if callable(closed_value) else closed_value)
        if kind in ("CIRCLE", "ELLIPSE"):
            closed = True
        if points:
            yield "line", points, closed


def convert_dwg(path: Path, converter: str | None, temp_dir: Path) -> Path:
    exe = converter or shutil.which("ODAFileConverter") or shutil.which("ODAFileConverter.exe")
    if not exe:
        raise ValueError("DWG input requires ODA File Converter; install it or pass --dwg-converter")
    source_dir, output_dir = temp_dir / "dwg_source", temp_dir / "dxf_output"
    source_dir.mkdir(); output_dir.mkdir()
    copied = source_dir / path.name
    shutil.copy2(path, copied)
    command = [exe, str(source_dir), str(output_dir), "ACAD2018", "DXF", "0", "1", "*.dwg"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ValueError(f"Could not launch ODA File Converter: {exc}") from exc
    converted = output_dir / (path.stem + ".dxf")
    if result.returncode != 0 or not converted.exists():
        details = (result.stderr or result.stdout).strip()
        raise ValueError(f"DWG to DXF conversion failed (exit {result.returncode}): {details}")
    return converted


def input_crs(path: Path, requested_crs: str) -> CRS:
    if path.suffix.lower() == ".shp":
        prj = path.with_suffix(".prj")
        return CRS.from_wkt(prj.read_text(encoding="utf-8", errors="replace")) if prj.exists() else UTM32N
    if requested_crs.strip().lower() != "auto":
        return CRS.from_user_input(requested_crs)

    coords = [pt for _, points, _ in iter_cad_features(path) for pt in points]
    if not coords:
        raise ValueError(f"No supported geometry found in {path.name}")
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # UTM eastings are below one million; GK zone 3 uses a 3,500,000 m false easting.
    if 100_000 <= xmin and xmax <= 900_000 and -1_000_000 <= ymin and ymax <= 10_000_000:
        return UTM32N
    if 3_000_000 <= xmin and xmax <= 4_000_000 and 4_000_000 <= ymin and ymax <= 7_000_000:
        return CRS.from_epsg(31467)
    raise ValueError(
        f"Cannot infer CRS from {path.name} bounds "
        f"({xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}); "
        "specify --input-crs, for example EPSG:25832 or EPSG:31467"
    )


def extract_tiles(archive: Path, destination: Path, bounds: tuple[float, float, float, float]) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        candidates = [(name, tile_bounds(name)) for name in zf.namelist()]
        # A tiny margin includes both tiles when geometry touches a tile edge.
        margin = 1e-6
        selection_bounds = (bounds[0] - margin, bounds[1] - margin,
                            bounds[2] + margin, bounds[3] + margin)
        selected = [(name, b) for name, b in candidates if b and intersect(selection_bounds, b)]
        if not selected:
            raise FileNotFoundError(f"No matching DGM1 tiles in {archive.name} cover {bounds}")
        for member, _ in selected:
            # Ignore directory traversal paths from untrusted ZIP members.
            target = destination / PurePosixPath(member).name
            with zf.open(member) as src, target.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            extracted.append(target)
    return extracted


def load_dem(paths: list[Path], bounds: tuple[float, float, float, float]):
    datasets = [rasterio.open(p) for p in paths]
    try:
        # Merge complete selected tiles so zero-area point bounds still produce
        # a usable raster window and edge coordinates remain sampleable.
        data, transform = merge(datasets, nodata=np.nan, dtype="float32", resampling=Resampling.nearest)
        return data[0], transform, datasets[0].crs
    finally:
        for ds in datasets:
            ds.close()


def sample_elevation(dem: np.ndarray, transform, x: float, y: float) -> float:
    col_f, row_f = (~transform) * (x, y)
    col, row = int(math.floor(col_f)), int(math.floor(row_f))
    if row < 0 or col < 0 or row >= dem.shape[0] or col >= dem.shape[1]:
        raise ValueError(f"Coordinate ({x:.3f}, {y:.3f}) falls outside extracted DGM coverage")
    value = float(dem[row, col])
    if not math.isfinite(value):
        raise ValueError(f"DGM has no elevation at ({x:.3f}, {y:.3f})")
    return value


def is_utm32n(crs) -> bool:
    """Accept EPSG:25832 and GeoTIFFs that store its name as a local CRS."""
    description = str(crs).upper()
    if "ETRS89 / UTM ZONE 32N" in description or "ETRS89 / UTM ZONE 32" in description:
        return True
    try:
        return CRS.from_user_input(crs).equals(UTM32N)
    except Exception:
        # Some installations pair a newer PROJ library with an older proj.db.
        return False


def export_dxf(source: Path, output: Path, dem: np.ndarray, transform, dem_crs: CRS, source_crs: CRS) -> int:
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6  # metres
    msp = doc.modelspace()
    to_dem = Transformer.from_crs(source_crs, dem_crs, always_xy=True) if source_crs != dem_crs else None
    count = 0

    def point(x: float, y: float) -> tuple[float, float, float]:
        if to_dem:
            x, y = to_dem.transform(x, y)
        return x, y, sample_elevation(dem, transform, x, y)

    if source.suffix.lower() == ".shp":
        reader = shapefile.Reader(str(source))
        try:
            for shape in reader.iterShapes():
                typ = shape.shapeType
                if typ == shapefile.NULL:
                    continue
                if typ in (shapefile.POINT, shapefile.POINTM, shapefile.POINTZ):
                    features = [("point", shape.points[:1], False)]
                elif typ in (shapefile.MULTIPOINT, shapefile.MULTIPOINTM, shapefile.MULTIPOINTZ):
                    features = [("point", [p], False) for p in shape.points]
                elif typ in (shapefile.POLYLINE, shapefile.POLYLINEM, shapefile.POLYLINEZ,
                             shapefile.POLYGON, shapefile.POLYGONM, shapefile.POLYGONZ):
                    starts = list(shape.parts) + [len(shape.points)]
                    closed = typ in (shapefile.POLYGON, shapefile.POLYGONM, shapefile.POLYGONZ)
                    features = [("line", shape.points[a:b], closed) for a, b in zip(starts, starts[1:])]
                else:
                    raise ValueError(f"Unsupported shapefile geometry type: {typ}")
                for kind, xy, closed in features:
                    pts = [point(x, y) for x, y, *_ in xy]
                    if kind == "point" and pts:
                        msp.add_point(pts[0])
                    elif len(pts) >= 2:
                        msp.add_polyline3d(pts, close=closed)
                count += 1
        finally:
            reader.close()
    else:
        for kind, xy, closed in iter_cad_features(source):
            pts = [point(x, y) for x, y in xy]
            if kind == "point":
                msp.add_point(pts[0])
            elif len(pts) >= 2:
                msp.add_polyline3d(pts, close=closed)
            count += 1
    doc.saveas(output)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="Input .shp, .dxf, or .dwg")
    parser.add_argument("dgm_zip", type=Path, help="ZIP containing dgm1_32_<east-km>_<north-km>_1_he.tif tiles")
    parser.add_argument("output_dxf", type=Path, help="Output DXF path")
    parser.add_argument("--work-dir", type=Path, help="Directory for extracted tiles (default: <output stem>_tiles)")
    parser.add_argument("--input-crs", default="auto", help="CRS for DXF/DWG coordinates: auto-detect UTM 32N or GK zone 3, or specify an EPSG code (default: auto)")
    parser.add_argument("--dwg-converter", help="ODA File Converter executable (or set it on PATH) for DWG input")
    args = parser.parse_args()

    source = args.input_file.resolve()
    if source.suffix.lower() not in (".shp", ".dxf", ".dwg"):
        parser.error("input_file must have a .shp, .dxf, or .dwg extension")
    try:
        cad_crs = input_crs(source, args.input_crs)
        with tempfile.TemporaryDirectory(prefix="drape_shape_") as temp:
            working_source = convert_dwg(source, args.dwg_converter, Path(temp)) if source.suffix.lower() == ".dwg" else source
            bounds = read_shape_bounds(working_source, UTM32N, cad_crs)
            work = args.work_dir or args.output_dxf.with_name(args.output_dxf.stem + "_tiles")
            tiles = extract_tiles(args.dgm_zip, work, bounds)
            dem, transform, dem_crs = load_dem(tiles, bounds)
            if dem_crs is None:
                raise ValueError("DGM tiles have no CRS metadata")
            if not is_utm32n(dem_crs):
                raise ValueError(f"Expected DGM in UTM zone 32N (EPSG:25832), got {dem_crs}")
            # Normalize the tile's named local CRS to avoid PROJ operations against LOCAL_CS.
            dem_crs = UTM32N
            n = export_dxf(working_source, args.output_dxf, dem, transform, dem_crs, cad_crs)
        print(f"Wrote {n} input features to {args.output_dxf}; extracted {len(tiles)} DGM tile(s) to {work}")
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, rasterio.errors.RasterioError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
