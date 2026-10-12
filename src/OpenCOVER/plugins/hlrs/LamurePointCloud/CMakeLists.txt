# Set the source directory
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Include additional CMake scripts
include(${SRC_DIR}/pcl_makros.cmake)

# Using the LamurePointCloud target
using(Lamure)
using(GLFW)

covise_find_package(SCHISM)
if(NOT SCHISM_FOUND)
    return()
endif()
include_directories(
    ${SCHISM_INCLUDE_DIRS}
)

# Include directories for Lamure
set(LAMURE_INCLUDE_DIRS
    #${SRC_DIR}/lamure/include
    ${SRC_DIR}/rendering/include
    ${SRC_DIR}/pvs_common/include
    ${SRC_DIR}/common/include
)

include_directories(
    ${LAMURE_INCLUDE_DIRS}
)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    policy.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     policy.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    bvh.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     bvh.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    bvh_stream.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     bvh_stream.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    controller.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     controller.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    gpu_context.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     gpu_context.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    gpu_cache.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     gpu_cache.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    gpu_access.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     gpu_access.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    ooc_pool.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     ooc_pool.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    ooc_cache.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     ooc_cache.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    cut.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     cut.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    cut_update_pool.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     cut_update_pool.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    cut_update_queue.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     cut_update_queue.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    cut_update_index.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     cut_update_index.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    cut_database.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     cut_database.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    cut_database_record.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     cut_database_record.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    cache_index.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     cache_index.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    cache_queue.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     cache_queue.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    cache.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     cache.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    camera.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     camera.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    trackball.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     trackball.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    dataset.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     dataset.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    model_database.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     model_database.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    lod_stream.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     lod_stream.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/rendering/src                                    provenance_stream.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/rendering/include/lamure/ren                     provenance_stream.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    pvs_database.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     pvs_database.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_bounding.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_bounding.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_regular.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_regular.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_regular_compressed.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_regular_compressed.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_irregular.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_irregular.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_irregular_compressed.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_irregular_compressed.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_octree.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_octree.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_octree_compressed.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_octree_compressed.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_octree_node.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_octree_node.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_octree_hierarchical_node.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_octree_hierarchical_node.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_octree_hierarchical.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_octree_hierarchical.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_octree_hierarchical_v2.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_octree_hierarchical_v2.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    grid_octree_hierarchical_v3.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     grid_octree_hierarchical_v3.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    view_cell_regular.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     view_cell_regular.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    view_cell_regular_managing.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     view_cell_regular_managing.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/pvs_common/src                                    view_cell_irregular.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/pvs_common/include/lamure/pvs                     view_cell_irregular.h)

pcl_project_files(SOURCE_FILES      ${SRC_DIR}/common/src                                    semaphore.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/common/include/lamure                     semaphore.h)
pcl_project_files(SOURCE_FILES      ${SRC_DIR}/common/src                                    bounding_box.cpp)
pcl_project_files(HEADER_FILES      ${SRC_DIR}/common/include/lamure                     bounding_box.h)

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
  LamureEditTool.h
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
  LamureEditTool.cpp
)


# Adding the LamurePointCloud plugin
cover_add_plugin(LamurePointCloud)

covise_find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(LamurePointCloud OpenMP::OpenMP_CXX)
endif()

covise_wnoerror(LamurePointCloud)
target_link_libraries(LamurePointCloud ${SCHISM_LIBRARIES})

