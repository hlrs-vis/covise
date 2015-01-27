IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

SUBDIRS(
        src
        applications
)
