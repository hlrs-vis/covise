#OpenCOVER plugin to load Google 3D tiles
USING(CURL)
USING(PROJ4)

SET(HEADERS
  3DTilesPlugin.h
)

SET(SOURCES
  3DTilesPlugin.cpp
)

cover_add_plugin(3DTiles  ${HEADERS})


target_include_directories(3DTiles PRIVATE ${PROJECT_SOURCE_DIR}/../../src/3rdparty/visionaray/src/3rdparty/rapidjson/include ${CURL_INCLUDE_DIR})

#COVISE_WNOERROR(3DTiles)
