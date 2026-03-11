option(BUILD_LOGGING "Build Logging-Lib" OFF)

if (NOT BUILD_LOGGING)
    message(STATUS "Logging with spdlog build is disabled. Set BUILD_LOGGING to ON to enable.")
    return()
endif()

find_package(spdlog QUIET CONFIG)

if (NOT spdlog_FOUND)
    message(STATUS "spdlog not found. Fetching and building from source.")
    include(FetchContent)
    
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
        GIT_TAG "v1.17.0"
        GIT_SHALLOW TRUE #copy version and not git history
    )
    
    FetchContent_MakeAvailable(spdlog)
else()
    message(STATUS "Found pre-installed spdlog. Using imported targets.")
endif()

# set(SOURCES
# )

set(HEADERS
    log.h
    export.h
)

set(Name "coLog")

ADD_COVISE_LIBRARY(${Name} ${HEADERS})
target_link_libraries(${Name} spdlog)
COVISE_INSTALL_TARGET(${Name})
COVISE_INSTALL_HEADERS(${Name} ${HEADERS})
COVISE_WNOERROR(${Name} BASEARCH zebu)
set_target_properties(${Name} PROPERTIES LINKER_LANGUAGE CXX)
