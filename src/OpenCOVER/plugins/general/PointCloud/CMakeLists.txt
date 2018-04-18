#OpenCOVER plugins calit2 PointCloud
USING(E57 optional)

SET(HEADERS
  PointCloud.h
  PointCloudGeometry.h
  PointCloudInteractor.h
)

SET(SOURCES
  PointCloud.cpp
  PointCloudGeometry.cpp
  PointCloudInteractor.cpp
)

cover_add_plugin(PointCloud)

SET(HEADERS
)
SET(SOURCES
  PointSort/PointConvert.cpp
)

ADD_COVISE_EXECUTABLE(PointConvert)
COVISE_WNOERROR(PointConvert)

TARGET_LINK_LIBRARIES(PointConvert ${EXTRA_LIBS} ${OPENSCENEGRAPH_LIBRARIES} )
SET(HEADERS
)
SET(SOURCES
  PointSort/PointSort.cpp
)

ADD_COVISE_EXECUTABLE(PointSort)
COVISE_WNOERROR(PointSort)

TARGET_LINK_LIBRARIES(PointSort ${EXTRA_LIBS} ${OPENSCENEGRAPH_LIBRARIES} )

IF(UNIX OR MINGW)
      SET(WITH_OPENMP "TRUE")
      IF(MINGW)
         SET(WITH_OPENMP "FALSE")
      ENDIF()
      IF(APPLE)
         # with cmake 2.8.8 (rc for now), clang will be identifiable
         # using CMAKE_COMPILER_ID. For now, use this hack to identify
         # clang on Max OS X
         EXEC_PROGRAM(${CMAKE_CXX_COMPILER} ARGS --version OUTPUT_VARIABLE _clang_version_info)
         IF(${_clang_version_info} MATCHES "clang")
            SET(WITH_OPENMP "FALSE")
         ENDIF(${_clang_version_info} MATCHES "clang")
      ENDIF(APPLE)

      IF(WITH_OPENMP)
         ADD_COVISE_COMPILE_FLAGS(PointSort "-fopenmp")
         ADD_COVISE_LINK_FLAGS(PointSort "-fopenmp")
      ENDIF()
   ENDIF(UNIX OR MINGW)

   IF(MSVC)
      ADD_COVISE_COMPILE_FLAGS(PointSort "/openmp")
   ENDIF(MSVC)

