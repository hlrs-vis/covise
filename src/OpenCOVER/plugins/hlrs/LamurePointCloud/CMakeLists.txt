# Set the source directory
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Include additional CMake scripts
include(${SRC_DIR}/pcl_makros.cmake)

# Using the LamurePointCloud target
MESSAGE("0")
using(Lamure)
MESSAGE("01")
using(FREEIMAGE)
MESSAGE("02")
using(GLUT)
MESSAGE("1")
using(GLFW)
MESSAGE("2")
using(FREETYPE)
MESSAGE("3")

# Include directories for Lamure
set(LAMURE_INCLUDE_DIRS
    ${SRC_DIR}/lamure/include
    ${SRC_DIR}/virtual_texturing/include
    ${SRC_DIR}/rendering/include
    ${SRC_DIR}/pvs_common/include
    ${SRC_DIR}/common/include
    ${SRC_DIR}/pvs_preprocessing/include
    ${SRC_DIR}/provenance/include
    ${SRC_DIR}/quality_measurement/include
)

include_directories(
    ${LAMURE_INCLUDE_DIRS}
)


pcl_project_files(SOURCE_FILES      ${SRC_DIR}/virtual_texturing/src                            *.cpp)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/virtual_texturing/src/ext                        *.cpp)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/virtual_texturing/src/ooc                        *.cpp)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/virtual_texturing/src/pre                        *.cpp)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/virtual_texturing/src/ren                        *.cpp)

pcl_project_files(HEADER_FILES      ${SRC_DIR}/virtual_texturing/include/lamure/vt              *.h)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/virtual_texturing/include/lamure/vt/ext          *.h)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/virtual_texturing/include/lamure/vt/ooc          *.h)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/virtual_texturing/include/lamure/vt/pre          *.h)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/virtual_texturing/include/lamure/vt/ren          *.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    *.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     *.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                   *.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                    *.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/common/src                                       *.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/common/include/lamure                            *.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_preprocessing/src                            *.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_preprocessing/include/lamure/pvs             *.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/provenance/src                                   *.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/provenance/include/lamure/prov                   *.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/quality_measurement/src                          *.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/quality_measurement/include/lamure/qm            *.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/lamure                                           *.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/lamure						                    *.h)

SET(HEADERS 
  ${HEADER_FILES}
  gl_state.h
  osg_util.h
  Lamure.h
  LamureRenderer.h
  LamureUI.h
  LamureUtil.h
  LamureMeasurement.h
  color_converter.h
)

SET(SOURCES 
  ${SOURCE_FILES}
  gl_state.cpp
  osg_util.cpp
  Lamure.cpp
  LamureRenderer.cpp
  LamureUI.cpp
  LamureUtil.cpp
  LamureMeasurement.cpp
  color_converter.cpp
)


# Adding the LamurePointCloud plugin
cover_add_plugin(LamurePointCloud)
covise_wnoerror(LamurePointCloud)

