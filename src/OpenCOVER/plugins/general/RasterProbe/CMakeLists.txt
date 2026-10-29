USING(OpenSceneGraph)
USING(VRML)
USING(GDAL)

set(HEADERS
    RasterProbe.h
    VrmlNodeRasterProbe.h
)

set(SOURCES
    RasterProbe.cpp
    VrmlNodeRasterProbe.cpp
)

cover_add_plugin(RasterProbe coVRGeoData)
