# - Find VolPack
# Find the VolPack includes and library
# This module defines
#  VOLPACK_INCLUDE_DIR, where to find volpack.h
#  VOLPACK_LIBRARIES, the libraries needed to use VolPack
#  VOLPACK_FOUND, If false, do not try to use VolPack.
# also defined, but not for general use are
#  VOLPACK_LIBRARY, where to find the VolPack library.

FIND_PATH(VOLPACK_INCLUDE_DIR volpack.h)
FIND_LIBRARY(VOLPACK_LIBRARY NAMES volpack)

# handle the QUIETLY and REQUIRED arguments and set VOLPACK_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(VOLPACK DEFAULT_MSG VOLPACK_LIBRARY VOLPACK_INCLUDE_DIR)

IF(VOLPACK_FOUND)
  SET(VOLPACK_LIBRARIES ${VOLPACK_LIBRARY})
ENDIF(VOLPACK_FOUND)

MARK_AS_ADVANCED(VOLPACK_LIBRARY VOLPACK_INCLUDE_DIR)
