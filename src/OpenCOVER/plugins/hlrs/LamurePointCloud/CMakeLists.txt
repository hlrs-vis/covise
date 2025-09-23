# Set the source directory
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Include additional CMake scripts
include(${SRC_DIR}/pcl_makros.cmake)

# Using the LamurePointCloud target
using(LamurePointCloud)

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
  #LamurePointCloud.h
  #LamurePointCloudInteractor.h
  #LamureGeometry.h
  #LamureDrawable.h
  gl_state.h
  osg_util.h
  Lamure.h
  LamureRenderer.h
  LamureUI.h
  LamureUtil.h
  LamureMeasurement.h
)

SET(SOURCES 
  ${SOURCE_FILES}
  #LamurePointCloud.cpp
  #LamurePointCloudInteractor.cpp
  #LamureGeometry.cpp
  #LamureDrawable.cpp
  gl_state.cpp
  osg_util.cpp
  Lamure.cpp
  LamureRenderer.cpp
  LamureUI.cpp
  LamureUtil.cpp
  LamureMeasurement.cpp
)

set(SCHISM_INCLUDE_DIRS
    $ENV{EXTERNLIBS}/../../schism
    $ENV{EXTERNLIBS}/../../schism/scm_core/src
    $ENV{EXTERNLIBS}/../../schism/scm_gl_core/src
    $ENV{EXTERNLIBS}/../../schism/scm_gl_util/src
    $ENV{EXTERNLIBS}/../../schism/scm_input/src
    $ENV{EXTERNLIBS}/../../schism/scm_cl_core/src
)
 
set(FREEGLUT_INCLUDE_DIR $ENV{EXTERNLIBS}/freeglut/include)
set(FREEIMAGE_INCLUDE_DIR $ENV{EXTERNLIBS}/FreeImage/include)
set(FREEIMAGEPLUS_INCLUDE_DIR $ENV{EXTERNLIBS}/FreeImage/include)
set(FREETYPE_INCLUDE_DIR $ENV{EXTERNLIBS}/freetype/include)
set(GLEW_INCLUDE_DIR $ENV{EXTERNLIBS}/glew/include)
set(GLFW_INCLUDE_DIR $ENV{EXTERNLIBS}/glfw/include)

# Adding include directories to the project
include_directories(
    ${FREEGLUT_INCLUDE_DIR}
    ${GLM_INCLUDE_DIR}  # Ensure GLM_INCLUDE_DIR is defined elsewhere
    ${FREETYPE_INCLUDE_DIR}
    ${FREEIMAGE_INCLUDE_DIR}
    ${FREEIMAGEPLUS_INCLUDE_DIR}
    ${LAMURE_INCLUDE_DIRS}
    ${SCHISM_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIR}
)

# Library paths
set(FREEGLUT_LIBRARY $ENV{EXTERNLIBS}/freeglut/lib/x64/freeglut.lib)

# FreeImage libraries
set(FREEIMAGE_LIBRARY_RELEASE $ENV{EXTERNLIBS}/FreeImage/lib/FreeImage.lib)
set(FREEIMAGE_LIBRARY_DEBUG $ENV{EXTERNLIBS}/FreeImage/lib/FreeImaged.lib)
set(FREEIMAGEPLUS_LIBRARY_RELEASE $ENV{EXTERNLIBS}/FreeImage/lib/FreeImagePlus.lib)
set(FREEIMAGEPLUS_LIBRARY_DEBUG $ENV{EXTERNLIBS}/FreeImage/lib/FreeImagePlusd.lib)

# FreeType libraries
set(FREETYPE_LIBRARY_RELEASE $ENV{EXTERNLIBS}/freetype/lib/freetype.lib)
set(FREETYPE_LIBRARY_DEBUG $ENV{EXTERNLIBS}/freetype/lib/freetyped.lib)

# Schism libraries
set(SCHISM_CORE_LIBRARY_RELEASE $ENV{EXTERNLIBS}/../../schism/lib/win_x64/Release/scm_core.lib)
set(SCHISM_CORE_LIBRARY_DEBUG $ENV{EXTERNLIBS}/../../schism/lib/win_x64/Debug/scm_core-gd.lib)
set(SCHISM_GL_CORE_LIBRARY_RELEASE $ENV{EXTERNLIBS}/../../schism/lib/win_x64/Release/scm_gl_core.lib)
set(SCHISM_GL_CORE_LIBRARY_DEBUG $ENV{EXTERNLIBS}/../../schism/lib/win_x64/Debug/scm_gl_core-gd.lib)
set(SCHISM_GL_UTIL_LIBRARY_RELEASE $ENV{EXTERNLIBS}/../../schism/lib/win_x64/Release/scm_gl_util.lib)
set(SCHISM_GL_UTIL_LIBRARY_DEBUG $ENV{EXTERNLIBS}/../../schism/lib/win_x64/Debug/scm_gl_util-gd.lib)

# GLEW and GLFW libraries
set(GLEW_LIBRARY $ENV{EXTERNLIBS}/glew/lib/glew32.lib)
set(GLFW_LIBRARY_RELEASE $ENV{EXTERNLIBS}/glfw/lib/glfw3.lib)
set(GLFW_LIBRARY_DEBUG $ENV{EXTERNLIBS}/glfw/lib/glfw3d.lib)

# Adding the LamurePointCloud plugin
cover_add_plugin(LamurePointCloud)
covise_wnoerror(LamurePointCloud)

# Linking libraries to the target
target_link_libraries(LamurePointCloud
    optimized ${FREEIMAGEPLUS_LIBRARY_RELEASE} 
    debug ${FREEIMAGEPLUS_LIBRARY_DEBUG}

    optimized ${GLEW_LIBRARY}
    debug ${GLEW_LIBRARY}

    optimized ${FREETYPE_LIBRARY_RELEASE}
    debug ${FREETYPE_LIBRARY_DEBUG}

    optimized ${GLFW_LIBRARY_RELEASE} 
    debug ${GLFW_LIBRARY_DEBUG}

    optimized ${SCHISM_CORE_LIBRARY_RELEASE} 
    debug ${SCHISM_CORE_LIBRARY_DEBUG}

    optimized ${SCHISM_GL_CORE_LIBRARY_RELEASE}
    debug ${SCHISM_GL_CORE_LIBRARY_DEBUG}

    optimized ${SCHISM_GL_UTIL_LIBRARY_RELEASE} 
    debug ${SCHISM_GL_UTIL_LIBRARY_DEBUG}

    optimized ${Boost_TIMER_LIBRARY_RELEASE} 
    debug ${Boost_TIMER_LIBRARY_DEBUG}

    optimized ${FREEIMAGE_LIBRARY_RELEASE} 
    debug ${FREEIMAGE_LIBRARY_DEBUG}

    optimized ${FREEGLUT_LIBRARY}
    debug ${FREEGLUT_LIBRARY}
)

# Display some environment variables for debugging
message("")
message("ENV{EXTERNLIBS}: $ENV{EXTERNLIBS}")
message("ENV{ARCHSUFFIX}: $ENV{ARCHSUFFIX}")
message("SRC_DIR: ${SRC_DIR}")
message("")

