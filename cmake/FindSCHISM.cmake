##############################################################################
# search paths
##############################################################################
SET(SCHISM_INCLUDE_SEARCH_DIRS
  $ENV{EXTERNLIBS}/schism/include
  ${GLOBAL_EXT_DIR}/schism/include
  ${GLOBAL_EXT_DIR}/include/schism
  ${SCHISM_INCLUDE_SEARCH_DIR}
  /opt/schism/current
)

SET(SCHISM_LIBRARY_SEARCH_DIRS
  $ENV{EXTERNLIBS}/schism/lib
  ${GLOBAL_EXT_DIR}/lib
  ${GLOBAL_EXT_DIR}/schism/lib
  ${SCHISM_LIBRARY_SEARCH_DIR}
  ../
  /opt/schism/current/lib/linux_x86
)

##############################################################################
# check for schism
##############################################################################

    MESSAGE("SEARCHING")
        FIND_PATH(SCHISM_INCLUDE_DIR
                NAMES scm_gl_core/src/scm/gl_core.h
                PATHS ${SCHISM_INCLUDE_SEARCH_DIRS}
                NO_DEFAULT_PATH)
        LIST(APPEND SCHISM_INCLUDE_DIRS ${SCHISM_INCLUDE_DIR}/scm_cl_core/src)
        LIST(APPEND SCHISM_INCLUDE_DIRS ${SCHISM_INCLUDE_DIR}/scm_core/src)
        LIST(APPEND SCHISM_INCLUDE_DIRS ${SCHISM_INCLUDE_DIR}/scm_gl_core/src)
        LIST(APPEND SCHISM_INCLUDE_DIRS ${SCHISM_INCLUDE_DIR}/scm_gl_util/src)
        LIST(APPEND SCHISM_INCLUDE_DIRS ${SCHISM_INCLUDE_DIR}/scm_input/src)
 
find_library(SCHISM_CORE_LIBRARY 
             NAMES scm_core libscm_core
             PATHS ${SCHISM_LIBRARY_SEARCH_DIRS}
             SUFFIXES release
            )

find_library(SCHISM_GL_CORE_LIBRARY 
             NAMES scm_gl_core libscm_gl_core
             PATHS ${SCHISM_LIBRARY_SEARCH_DIRS}
             SUFFIXES release
            )

find_library(SCHISM_GL_UTIL_LIBRARY 
             NAMES scm_gl_util libscm_gl_util
             PATHS ${SCHISM_LIBRARY_SEARCH_DIRS}
             SUFFIXES release
            )

find_library(SCHISM_INPUT_LIBRARY 
             NAMES scm_input libscm_input
             PATHS ${SCHISM_LIBRARY_SEARCH_DIRS}
             SUFFIXES release
            )

# find debug libraries
find_library(SCHISM_CORE_LIBRARY_DEBUG
             NAMES scm_core-gd libscm_core-gd scm_core libscm_core
             PATHS ${SCHISM_LIBRARY_SEARCH_DIRS}
             SUFFIXES debug
            )

find_library(SCHISM_GL_CORE_LIBRARY_DEBUG
             NAMES scm_gl_core-gd libscm_gl_core-gd scm_gl_core libscm_gl_core
             PATHS ${SCHISM_LIBRARY_SEARCH_DIRS}
             SUFFIXES debug
            )

find_library(SCHISM_GL_UTIL_LIBRARY_DEBUG
             NAMES scm_gl_util-gd libscm_gl_util-gd scm_gl_util libscm_gl_util
             PATHS ${SCHISM_LIBRARY_SEARCH_DIRS}
             SUFFIXES debug
            )

find_library(SCHISM_INPUT_LIBRARY_DEBUG
             NAMES scm_input-gd libscm_input-gd scm_input libscm_input
             PATHS ${SCHISM_LIBRARY_SEARCH_DIRS}
             SUFFIXES debug
            )
			
			
IF(MSVC)
  
  IF(SCHISM_CORE_LIBRARY_DEBUG AND SCHISM_CORE_LIBRARY)
    SET(SCHISM_LIBRARIES optimized ${SCHISM_CORE_LIBRARY} debug ${SCHISM_CORE_LIBRARY_DEBUG} optimized ${SCHISM_GL_CORE_LIBRARY} debug ${SCHISM_GL_CORE_LIBRARY_DEBUG} optimized ${SCHISM_GL_UTIL_LIBRARY} debug ${SCHISM_GL_UTIL_LIBRARY_DEBUG} optimized ${SCHISM_INPUT_LIBRARY} debug ${SCHISM_INPUT_LIBRARY_DEBUG})
  ENDIF(SCHISM_CORE_LIBRARY_DEBUG AND SCHISM_CORE_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(SCHISM DEFAULT_MSG SCHISM_CORE_LIBRARY SCHISM_CORE_LIBRARY_DEBUG SCHISM_GL_CORE_LIBRARY SCHISM_GL_CORE_LIBRARY_DEBUG SCHISM_GL_UTIL_LIBRARY SCHISM_GL_UTIL_LIBRARY_DEBUG SCHISM_INPUT_LIBRARY SCHISM_INPUT_LIBRARY_DEBUG SCHISM_INCLUDE_DIR)

  MARK_AS_ADVANCED(SCHISM_CORE_LIBRARY SCHISM_GL_CORE_LIBRARY SCHISM_GL_UTIL_LIBRARY SCHISM_INPUT_LIBRARY)
  
ELSE(MSVC)
  # rest of the world
    SET(SCHISM_LIBRARIES ${SCHISM_CORE_LIBRARY} ${SCHISM_GL_CORE_LIBRARY} ${SCHISM_GL_UTIL_LIBRARY} ${SCHISM_INPUT_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(SCHISM DEFAULT_MSG SCHISM_CORE_LIBRARY SCHISM_GL_CORE_LIBRARY SCHISM_GL_UTIL_LIBRARY SCHISM_INPUT_LIBRARY SCHISM_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(SCHISM_CORE_LIBRARY SCHISM_GL_CORE_LIBRARY SCHISM_GL_UTIL_LIBRARY SCHISM_INPUT_LIBRARY )
  
ENDIF(MSVC)
