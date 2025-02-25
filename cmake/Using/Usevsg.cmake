MACRO(USE_VSG)
  IF (((NOT vsgXchange_FOUND) OR (NOT vsg_FOUND) OR (NOT vsgPoints_FOUND))  AND (${ARGC} LESS 1))
    USING_MESSAGE("Skipping because of missing vsg")
    RETURN()
  ENDIF (((NOT vsgXchange_FOUND) OR (NOT vsg_FOUND) OR (NOT vsgPoints_FOUND))  AND (${ARGC} LESS 1))
  SET(vsg_USED TRUE)
  INCLUDE_DIRECTORIES(SYSTEM ${VSG_INCLUDE_DIRS})
  ADD_DEFINITIONS(-DHAVE_VSG)
  SET(EXTRA_LIBS ${EXTRA_LIBS} vsg::vsg vsgPoints::vsgPoints vsgXchange::vsgXchange)
ENDMACRO(USE_VSG)

