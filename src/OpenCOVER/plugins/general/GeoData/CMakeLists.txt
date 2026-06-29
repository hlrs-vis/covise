
USING(OpenSceneGraph)
USING(GDAL)
USING(PROJ)
USING(CURL)
USING(VRML)

covise_find_package(OpenSceneGraph 3.6.0 QUIET COMPONENTS osgTerrain)
COVISE_FIND_PACKAGE(osgTerrain)
IF(NOT OSGTERRAIN_FOUND)
  RETURN()
ENDIF()


set(HEADERS
    GeoDataLoader.h
    PlaceLabel.h
	VrmlNodeGeoData.h
)

set(SOURCES
    GeoDataLoader.cpp
    PlaceLabel.cpp
	VrmlNodeGeoData.cpp
)


cover_add_plugin(GeoData coCurlHTTPClient coVRGeoData)
target_include_directories(GeoData PRIVATE ${PROJECT_SOURCE_DIR}/../../src/3rdparty/visionaray/src/3rdparty/rapidjson/include)
