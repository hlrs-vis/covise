
using(OpenSceneGraph)
using(GDAL)
USING(PROJ)

covise_find_package(OpenSceneGraph 3.2.0 QUIET COMPONENTS osgTerrain)
COVISE_FIND_PACKAGE(osgTerrain)
IF(NOT OSGTERRAIN_FOUND)
  RETURN()
ENDIF()


set(HEADERS
    CutGeometry.h
    GeoDataLoader.h
)

set(SOURCES
    CutGeometry.cpp
    GeoDataLoader.cpp
)


cover_add_plugin(GeoData coCurlHTTPClient)
target_include_directories(GeoData PRIVATE ${PROJECT_SOURCE_DIR}/../../src/3rdparty/visionaray/src/3rdparty/rapidjson/include)
