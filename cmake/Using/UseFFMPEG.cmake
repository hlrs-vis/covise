MACRO(USE_FFMPEG)
  IF(NOT FFMPEG_USED)
    SET(FFMPEG_USED TRUE)
    covise_find_package(FFMPEG)
    if ((NOT FFMPEG_FOUND) AND (${ARGC} LESS 1))
      return()
    else()
      message(STATUS "Checking whether we are actually compiling against libav")
      try_compile(LIBAV_FOUND
        "${CMAKE_BINARY_DIR}/tmp"
        SOURCES
          "${COVISEDIR}/cmake/tests/libav.c"
        CMAKE_FLAGS
          "-DINCLUDE_DIRECTORIES=${FFMPEG_INCLUDE_DIRS}"
      )
      if (LIBAV_FOUND)
        add_definitions(-DHAVE_LIBAV)
        message(STATUS "  using libav, not all features will be supported")
      endif()
      ADD_DEFINITIONS(-DHAVE_FFMPEG)
      INCLUDE_DIRECTORIES(${FFMPEG_INCLUDE_DIRS})
      SET(EXTRA_LIBS ${EXTRA_LIBS} ${FFMPEG_LIBRARIES})
      if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        SET(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations")
      elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
      endif()
    endif()
  ENDIF(NOT FFMPEG_USED)
ENDMACRO(USE_FFMPEG)
