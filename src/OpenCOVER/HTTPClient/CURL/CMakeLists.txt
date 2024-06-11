USING(CURL)
if(NOT CURL_FOUND)
    message("CURL not found, not building HTTPClient library")
    return()
endif()

file(GLOB_RECURSE SOURCES "*.cpp")
# file(GLOB_RECURSE SOURCES "*.cpp" "*.h")
file(GLOB_RECURSE HEADERS "*.h")
# add_library(CurlHTTPClient OBJECT ${SOURCES})
ADD_COVISE_LIBRARY(coCurlHTTPClient ${SOURCES})

target_link_libraries(coCurlHTTPClient ${CURL_LIBRARIES})
COVISE_INSTALL_TARGET(coCurlHTTPClient)
COVISE_INSTALL_HEADERS(coCurlHTTPClient ${HEADERS})

# TARGET_INCLUDE_DIRECTORIES(coOpenOpcUaClient PRIVATE ${OPENSCENEGRAPH_INCLUDE_DIRS})

COVISE_WNOERROR(coCurlHTTPClient BASEARCH zebu)