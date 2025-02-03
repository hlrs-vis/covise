USING(GDAL)
USING(CURL)

SET(HEADERS
  UrbanTempo.h
  TreeModel.h 
  Tree.h
)
SET(SOURCES
  UrbanTempo.cpp
  TreeModel.cpp 
  Tree.cpp
)

cover_add_plugin(UrbanTempo ${HEADERS} ${SOURCES} coCurlHTTPClient)
target_include_directories(UrbanTempo PRIVATE ${PROJECT_SOURCE_DIR}/../../src/3rdparty/visionaray/src/3rdparty/rapidjson/include)
