USING(CURL)
if(NOT CURL_FOUND)
    message("CURL not found, not building HTTPClient library")
    return()
endif()

file(GLOB_RECURSE SOURCES "*.cpp")
file(GLOB_RECURSE HEADERS "*.h")
ADD_COVISE_LIBRARY(coCurlHTTPClient ${SOURCES})

target_link_libraries(coCurlHTTPClient ${CURL_LIBRARIES})
COVISE_INSTALL_TARGET(coCurlHTTPClient)
COVISE_INSTALL_HEADERS(coCurlHTTPClient ${HEADERS})

COVISE_WNOERROR(coCurlHTTPClient BASEARCH zebu)