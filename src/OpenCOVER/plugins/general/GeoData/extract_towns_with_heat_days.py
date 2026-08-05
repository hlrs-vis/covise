#!/usr/bin/env python3
"""
This script extracts place information from OSM data, assigns an altitude from
a heightmap, and saves the result as GeoJSON.

Places, such as towns and cities, are filtered by their category tag (see
PLACES below) and assigned a numerical size property. A lookup in the specified
heightmap file returns the altitude of the place. Only named places are processed.
The result can be read by the GeoData plugin as place labels.

You can download DGM200 for Germany here (the .asc file type is directly supported):

    https://gdz.bkg.bund.de/index.php/default/digitales-gelandemodell-gitterweite-200-m-dgm200.html
"""

import argparse

from numpy import number
import osmium
import osmium.filter
from osmium.osm import Node
import rasterio
from rasterio.warp import transform
from rasterio.transform import rowcol
from geojson import FeatureCollection, Feature, Point, dump

PLACES = {"city": 3, "town": 2, "village": 1, "suburb": 1}


def height_at(img, band, lon, lat):
    (x,), (y,), *_ = transform("EPSG:4326", img.crs, [lon], [lat])
    row, col = rowcol(img.transform, x, y)
    if row < 0 or col < 0 or row >= band.shape[0] or col >= band.shape[1]:
        return None
    value = band[row, col]
    nodata = img.nodatavals[0]
    return None if nodata is not None and value == nodata else float(value)

def heatdays_at(img, band, lon, lat):
    (x,), (y,), *_ = transform("EPSG:4326", img.crs, [lon], [lat])
    row, col = rowcol(img.transform, x, y)
    if row < 0 or col < 0 or row >= band.shape[0] or col >= band.shape[1]:
        return None
    value = band[row, col]
    nodata = img.nodatavals[0]
    return None if nodata is not None and value == nodata else float(value)

def main():
    parser = argparse.ArgumentParser(
        description="extract place information from OSM data, assign an altitude from a GeoTiff, and save as GeoJSON"
    )
    parser.add_argument("input_osm", help="a OSM extract file (*.osm.pbf)")
    parser.add_argument(
        "heightmap",
        help="a raster file of the altitudes in the desired height CRS (usually DHHN2016). Can be anything gdal can read (e. g. geotiff).",
    )
    parser.add_argument(
        "heatdays_present",
        help="a raster file with present heat day data. Can be anything gdal can read (e. g. geotiff).",
    )  
    parser.add_argument(
        "heatdays_future1",
        help="a raster file with future heat day data. Can be anything gdal can read (e. g. geotiff).",
    )      
    parser.add_argument(
        "heatdays_future2",
        help="a raster file with future heat day data. Can be anything gdal can read (e. g. geotiff).",
    )  
    parser.add_argument(
        "heatdays_future3",
        help="a raster file with future heat day data. Can be anything gdal can read (e. g. geotiff).",
    )  
    parser.add_argument("output_geojson", help="the output path for the GeoJSON file")
    args = parser.parse_args()

    points = []

    fp = (
        osmium.FileProcessor(args.input_osm)
        .with_filter(osmium.filter.KeyFilter("place"))
        .with_filter(osmium.filter.KeyFilter("name"))
    )

    number_of_twons_with_heat_data = 0
    with rasterio.open(args.heightmap) as img:
        band = img.read(1)
    with rasterio.open(args.heatdays_present) as heatdays_present_img:
        heatdays_present = heatdays_present_img.read(1)
    with rasterio.open(args.heatdays_future1) as heatdays_future1_img:
        heatdays_future1 = heatdays_future1_img.read(1)
    with rasterio.open(args.heatdays_future2) as heatdays_future2_img:
        heatdays_future2 = heatdays_future2_img.read(1) 
    with rasterio.open(args.heatdays_future3) as heatdays_future3_img:
        heatdays_future3 = heatdays_future3_img.read(1) 


        for obj in fp:
            place = obj.tags["place"]
            if isinstance(obj, Node) and place in PLACES:
                lonlat = (obj.lon, obj.lat)

                altitude = height_at(img, band, *lonlat)
                heatdays_present_value = heatdays_at(heatdays_present_img, heatdays_present, *lonlat)
                heatdays_future1_value = heatdays_at(heatdays_future1_img, heatdays_future1, *lonlat)
                heatdays_future2_value = heatdays_at(heatdays_future2_img, heatdays_future2, *lonlat)
                heatdays_future3_value = heatdays_at(heatdays_future3_img, heatdays_future3, *lonlat)

                tags = {
                    "name": obj.tags["name"],
                    "size": PLACES[place],
                }
                if altitude is not None:
                    tags["altitude"] = altitude

                if heatdays_present_value is not None:
                    number_of_twons_with_heat_data += 1
                    tags["name"] = tags["name"] + f", Hitzetage Ref: {heatdays_present_value:.1f},"
                if heatdays_future1_value is not None:
                    tags["name"] = tags["name"] + f" RCP4.5 2021: +{heatdays_future1_value:.1f},"
                if heatdays_future2_value is not None:
                    tags["name"] = tags["name"] + f" RCP4.5 2071: +{heatdays_future2_value:.1f},"
                if heatdays_future3_value is not None:
                    tags["name"] = tags["name"] + f" RCP8.5 2071: +{heatdays_future3_value:.1f}"

                points.append(Feature(geometry=Point(lonlat), properties=tags))

    with open(args.output_geojson, "wt") as f:
        dump(FeatureCollection(points), f, sort_keys=True, indent=2)
    print(f"Number of towns with heat data: {number_of_twons_with_heat_data}")

if __name__ == "__main__":
    main()
