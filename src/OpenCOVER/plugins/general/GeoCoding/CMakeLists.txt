USING(OpenSceneGraph)
USING(CURL)

set(HEADERS
    GeoCoding.h
)

set(SOURCES
    GeoCoding.cpp
)


cover_add_plugin(GeoCoding coCurlHTTPClient coVRGeoData)
target_include_directories(GeoCoding PRIVATE ${PROJECT_SOURCE_DIR}/../../src/3rdparty/visionaray/src/3rdparty/rapidjson/include)
