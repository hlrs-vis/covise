# - Find 3DSMAX
# Find the 3DSMAX includes and library
#
#  3DSMAX_INCLUDE_DIR - Where to find 3DSMAX includes
#  3DSMAX_LIBRARIES   - List of libraries when using 3DSMAX
#  3DSMAX_FOUND       - True if 3DSMAX was found

IF(3DSMAX_INCLUDE_DIR)
  SET(3DSMAX_FIND_QUIETLY TRUE)
ENDIF(3DSMAX_INCLUDE_DIR)


FIND_PATH(3DSMAX_STDPLUGS_DIR "3dsexp.dle"
  PATHS
  "C:/Program Files/Autodesk/3ds Max 2017/stdplugs"
  "C:/Program Files/Autodesk/3ds Max 2016/stdplugs"
  "C:/Program Files/Autodesk/3ds Max 2015/stdplugs"
  "C:/Program Files/Autodesk/3ds Max 2014/stdplugs"
  "C:/Program Files/Autodesk/3ds Max 2013/stdplugs"
  "C:/Program Files (x86)/Autodesk/3ds Max 2014/stdplugs"
  DOC "3DSMAX - plugins directory"
)

FIND_PATH(3DSMAX_INCLUDE_DIR "max.h"
  PATHS
  "C:/Program Files/Autodesk/3ds Max 2017 SDK/maxsdk/include"
  "C:/Program Files/Autodesk/3ds Max 2016 SDK/maxsdk/include"
  "C:/Program Files/Autodesk/3ds Max 2015 SDK/maxsdk/include"
  "C:/Program Files/Autodesk/3ds Max 2014 SDK/maxsdk/include"
  "C:/Program Files (x86)/Autodesk/3ds Max 2013 SDK/maxsdk/include"
  "C:/Program Files (x86)/Autodesk/maxsdk/include"
  $ENV{3DSMAXSDK}/include
  DOC "3DSMAX - Headers"
)

FIND_PATH(3DSMAX_SCRIPT_INCLUDE_DIR "maxscript.h"
  PATHS
  "C:/Program Files/Autodesk/3ds Max 2017 SDK/maxsdk/include/maxscript"
  "C:/Program Files/Autodesk/3ds Max 2016 SDK/maxsdk/include/maxscript"
  "C:/Program Files/Autodesk/3ds Max 2015 SDK/maxsdk/include/maxscript"
  "C:/Program Files/Autodesk/3ds Max 2014 SDK/maxsdk/include/maxscript"
  "C:/Program Files (x86)/Autodesk/3ds Max 2013 SDK/maxsdk/include/maxscript"
  "C:/Program Files (x86)/Autodesk/maxsdk/include/maxscript"
  $ENV{3DSMAXSDK}/include/maxscript
  DOC "3DSMAXSCRIPT - Headers"
)

FIND_PATH(3DSMAX_LIB_DIR "maxscrpt.lib"
  PATHS
  "C:/Program Files/Autodesk/3ds Max 2017 SDK/maxsdk/lib/x64/Release"
  "C:/Program Files/Autodesk/3ds Max 2016 SDK/maxsdk/lib/x64/Release"
  "C:/Program Files/Autodesk/3ds Max 2015 SDK/maxsdk/lib/x64/Release"
  "C:/Program Files/Autodesk/3ds Max 2014 SDK/maxsdk/lib/x64/Release"
  "C:/Program Files (x86)/Autodesk/3ds Max 2013 SDK/maxsdk/x64/lib"
  "C:/Program Files (x86)/Autodesk/maxsdk/x64/lib"
  $ENV{3DSMAXSDK}/x64/lib
  DOC "3DSMAX - Libraries"
)
SET(3DSMAX_LIBRARIES
comctl32.lib
version.lib
maxscrpt.lib
bmm.lib
core.lib
geom.lib
gfx.lib
maxutil.lib
mesh.lib
wsock32.lib
ws2_32.lib)
#helpsys.lib

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(x3DSMAX DEFAULT_MSG 3DSMAX_LIB_DIR 3DSMAX_SCRIPT_INCLUDE_DIR 3DSMAX_INCLUDE_DIR 3DSMAX_STDPLUGS_DIR)

  #MARK_AS_ADVANCED(3DSMAX_LIB_DIR 3DSMAX_SCRIPT_INCLUDE_DIR 3DSMAX_INCLUDE_DIR)
  
ELSE(MSVC)
    RETURN()
ENDIF(MSVC)

IF(3DSMAX_FOUND)
  SET(3DSMAX_INCLUDE_DIRS ${3DSMAX_INCLUDE_DIR} ${3DSMAX_SCRIPT_INCLUDE_DIR})
ELSE(3DSMAX_FOUND)
    RETURN()
ENDIF(3DSMAX_FOUND)
