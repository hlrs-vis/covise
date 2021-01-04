MACRO(USE_EMBREE3)
  covise_find_package(embree 3.0.0 QUIET)

  if (NOT embree_FOUND AND (${ARGC} LESS 1))
    using_message("Skipping because of missing EMBREE")
    return()
  endif()

  if(NOT EMBREE3_USED AND embree_FOUND)
    set(EMBREE3_USED TRUE)
    include_directories(SYSTEM ${EMBREE_INCLUDE_DIRS})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${EMBREE_LIBRARIES})
  endif()
ENDMACRO(USE_EMBREE3)
