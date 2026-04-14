set(CONFIG_SOURCES config.cpp)
set(CONFIG_HEADERS array.h access.h file.h value.h config.h)

set(PREFIX ../../OpenCOVER/OpenConfig/covconfig/)
include(${PREFIX}covconfig.cmake)

add_covise_library(vvOpenConfig ${CONFIG_SOURCES} ${CONFIG_HEADERS} ${COVCONFIG_SOURCES} ${COVCONFIG_HEADERS} ${COVCONFIG_DETAIL_HEADERS})
target_include_directories(vvOpenConfig SYSTEM PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_compile_definitions(vvOpenConfig PRIVATE CONFIG_NAMESPACE=vive COVCONFIG_VIVE)
target_include_directories(vvOpenConfig PRIVATE ${COVCONFIG_PRIVATE_INCLUDES})
target_link_libraries(vvOpenConfig ${COVCONFIG_PRIVATE_LIBRARIES})
covise_install_target(vvOpenConfig)
covise_install_headers(OpenConfig ${CONFIG_HEADERS})
