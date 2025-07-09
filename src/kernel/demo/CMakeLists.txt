covise_add_library(hlrsDemoCommon demo.cpp demo.h)
target_include_directories(hlrsDemoCommon
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/covise/kernel/demo>
)