# - Locate Open Inventor
#
# The file tries to locate Open Inventor library
#
# This module defines:
#  INVENTOR_FOUND, if false, do not try to link against Inventor.
#  INVENTOR_INCLUDE_DIR, where to find headers.
#  INVENTOR_LIBRARY, the library to link against.
#  INVENTOR_XT_LIBRARY, the Xt library - window binding library for Inventor
#

# try to find Inventor includes (regular paths)
FIND_PATH(INVENTOR_INCLUDE_DIR Inventor/Xt/SoXt.h
    $ENV{OIV_INCPATH}
    $ENV{EXTERNLIBS}/inventor/include
    $ENV{EXTERNLIBS}/OpenInventor/include
    /usr/local/include
    /usr/include
    /sw/include
    /opt/local/include
    /opt/csw/include
    /opt/include
)

# default Inventor lib search paths
SET(INVENTOR_LIB_SEARCH_PATH
    $ENV{OIV_LIBPATH}
    $ENV{EXTERNLIBS}/inventor/lib
    $ENV{EXTERNLIBS}/OpenInventor/lib
    /usr/local/lib
    /usr/lib
    /sw/lib
    /opt/local/lib
    /opt/csw/lib
    /opt/lib
    NO_DEFAULT_PATH
    NO_CMAKE_SYSTEM_PATH
)

# try to find SGI Inventor lib
IF(NOT INVENTOR_LIBRARY)
    FIND_LIBRARY(INVENTOR_LIBRARY
        NAMES Inventor
        PATHS ${INVENTOR_LIB_SEARCH_PATH}
    )
    FIND_LIBRARY(INVENTOR_XT_LIBRARY
        NAMES InventorXt
        PATHS ${INVENTOR_LIB_SEARCH_PATH}
    )
ENDIF(NOT INVENTOR_LIBRARY)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(INVENTOR DEFAULT_MSG
   INVENTOR_INCLUDE_DIR INVENTOR_LIBRARY INVENTOR_XT_LIBRARY)

IF(INVENTOR_FOUND)
  SET(INVENTOR_LIBRARIES ${INVENTOR_LIBRARY} ${INVENTOR_XT_LIBRARY})
ENDIF(INVENTOR_FOUND)

MARK_AS_ADVANCED(INVENTOR_LIBRARY INVENTOR_XT_LIBRARY INVENTOR_INCLUDE_DIR)
