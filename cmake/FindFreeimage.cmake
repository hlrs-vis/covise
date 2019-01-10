##############################################################################
# search paths
##############################################################################
SET(FREEIMAGE_INCLUDE_SEARCH_DIRS
    $ENV{EXTERNLIBS}/freeimage/include
    ${GLOBAL_EXT_DIR}/freeimage/include
    /usr/include
)

SET(FREEIMAGE_LIBRARY_SEARCH_DIRS
    $ENV{EXTERNLIBS}/freeimage/lib
    ${GLOBAL_EXT_DIR}/freeimage/lib
    /usr/lib
)

##############################################################################
# search include and library
##############################################################################
#  FREEIMAGE_FOUND - System has freeimage
#  FREEIMAGE_INCLUDE_DIRS - The freeimage include directories
#  FREEIMAGE_LIBRARIES - The libraries needed to use freeimage

IF (MSVC)
  SET(_FREEIMAGE_LIBRARY_NAME FreeImage.lib)
  SET(_FREEIMAGEPLUS_LIBRARY_NAME FreeImagePlus.lib)
ELSEIF (UNIX)
  SET(_FREEIMAGE_LIBRARY_NAME libfreeimage.so)
  SET(_FREEIMAGEPLUS_LIBRARY_NAME libfreeimageplus.so)
ENDIF (MSVC)

find_path(FREEIMAGE_INCLUDE_DIR
          NAMES FreeImagePlus.h
          PATHS ${FREEIMAGE_INCLUDE_SEARCH_DIRS}
         )

find_library(FREEIMAGE_LIBRARY
            NAMES ${_FREEIMAGE_LIBRARY_NAME}
            PATHS ${FREEIMAGE_LIBRARY_SEARCH_DIRS}
            PATH_SUFFIXES release
            )

find_library(FREEIMAGEPLUS_LIBRARY
            NAMES ${_FREEIMAGEPLUS_LIBRARY_NAME}
            PATHS ${FREEIMAGE_LIBRARY_SEARCH_DIRS}
            PATH_SUFFIXES release
            )


find_library(FREEIMAGE_LIBRARY_DEBUG
            NAMES ${_FREEIMAGE_LIBRARY_NAME}
            PATHS ${FREEIMAGE_LIBRARY_SEARCH_DIRS}
            PATH_SUFFIXES debug
            )

find_library(FREEIMAGEPLUS_LIBRARY_DEBUG
            NAMES ${_FREEIMAGEPLUS_LIBRARY_NAME}
            PATHS ${FREEIMAGE_LIBRARY_SEARCH_DIRS}
            PATH_SUFFIXES debug
            )

##############################################################################
# verify
##############################################################################
IF ( NOT FREEIMAGE_INCLUDE_DIR OR NOT FREEIMAGE_LIBRARY OR NOT FREEIMAGE_LIBRARY_DEBUG )
    SET(FREEIMAGE_INCLUDE_SEARCH_DIR "Please provide freeimage include path." CACHE PATH "path to freeimage headers.")
    SET(FREEIMAGE_LIBRARY_SEARCH_DIR "Please provide freeimage library path." CACHE PATH "path to freeimage libraries.")
    MESSAGE(STATUS "find_freeimage.cmake: unable to find freeimage library")
ELSE ( NOT FREEIMAGE_INCLUDE_DIR OR NOT FREEIMAGE_LIBRARY OR NOT FREEIMAGE_LIBRARY_DEBUG )
    UNSET(FREEIMAGE_INCLUDE_SEARCH_DIR CACHE)
    UNSET(FREEIMAGE_LIBRARY_SEARCH_DIR CACHE)
    MESSAGE(STATUS "--  found matching freeimage version")
ENDIF ( NOT FREEIMAGE_INCLUDE_DIR OR NOT FREEIMAGE_LIBRARY OR NOT FREEIMAGE_LIBRARY_DEBUG )
	
IF(MSVC)
  
  IF(FREEIMAGE_LIBRARY AND FREEIMAGE_LIBRARY_DEBUG)
    SET(FREEIMAGE_LIBRARIES optimized ${FREEIMAGE_LIBRARY} debug ${FREEIMAGE_LIBRARY_DEBUG} optimized ${FREEIMAGEPLUS_LIBRARY} debug ${FREEIMAGEPLUS_LIBRARY_DEBUG} )
  ENDIF(FREEIMAGE_LIBRARY AND FREEIMAGE_LIBRARY_DEBUG)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(FREEIMAGE DEFAULT_MSG FREEIMAGE_LIBRARY FREEIMAGE_LIBRARY_DEBUG FREEIMAGEPLUS_LIBRARY FREEIMAGEPLUS_LIBRARY_DEBUG FREEIMAGE_INCLUDE_DIR)

  MARK_AS_ADVANCED(FREEIMAGE_LIBRARY FREEIMAGE_LIBRARY_DEBUG FREEIMAGEPLUS_LIBRARY FREEIMAGEPLUS_LIBRARY_DEBUG)
  
ELSE(MSVC)
  # rest of the world
    SET(FREEIMAGE_LIBRARIES ${FREEIMAGE_LIBRARY} ${FREEIMAGEPLUS_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(FREEIMAGE DEFAULT_MSG FREEIMAGE_LIBRARY FREEIMAGEPLUS_LIBRARY FREEIMAGE_INPUT_LIBRARY FREEIMAGE_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(FREEIMAGE_LIBRARY FREEIMAGEPLUS_LIBRARY  )
  
ENDIF(MSVC)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FREEIMAGE DEFAULT_MSG FREEIMAGE_LIBRARY FREEIMAGEPLUS_LIBRARY FREEIMAGE_INCLUDE_DIR)
