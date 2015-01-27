#
# Try to find Cg
#
# CG_FOUND         = NVIDIA Cg was found
# CG_INCLUDE_DIR   = directory with cg.h
# CG_INCLUDE_DIRS  = same as above (not chached)
# CG_LIBRARY       = path to libCg
# CG_GL_LIBRARY    = path to libCgGL
# CG_COMPILER      = path to cgc
# 

IF(CG_INCLUDE_DIR)
  SET(CG_FIND_QUIETLY TRUE)
ENDIF(CG_INCLUDE_DIR)

# Decide between 32-bit and 64-bit binaries/libraries
IF(CMAKE_CL_64)
  SET(CG_LIBPATH "$ENV{CG_LIB64_PATH}")
  SET(CG_BINPATH "$ENV{CG_BIN64_PATH}")
  IF(EXISTS "$ENV{EXTERNLIBS}/cg")
    SET(CG_LIBPATH "$ENV{EXTERNLIBS}/cg/lib.x64")
    SET(CG_BINPATH "$ENV{EXTERNLIBS}/cg/bin.x64")
  ELSEIF(EXISTS "$ENV{EXTERNLIBS}/Cg")
    SET(CG_LIBPATH "$ENV{EXTERNLIBS}/Cg/lib.x64")
    SET(CG_BINPATH "$ENV{EXTERNLIBS}/Cg/bin.x64")
  ENDIF(EXISTS "$ENV{EXTERNLIBS}/cg")
ELSE(CMAKE_CL_64)
  SET(CG_LIBPATH "$ENV{CG_LIB_PATH}")
  SET(CG_LIBPATH "$ENV{CG_BIN_PATH}")
  IF(EXISTS "$ENV{EXTERNLIBS}/cg")
    SET(CG_LIBPATH "$ENV{EXTERNLIBS}/cg/lib")
    SET(CG_BINPATH "$ENV{EXTERNLIBS}/cg/bin" )
  ELSEIF(EXISTS "$ENV{EXTERNLIBS}/Cg")
    SET(CG_LIBPATH "$ENV{EXTERNLIBS}/Cg/lib")
    SET(CG_BINPATH "$ENV{EXTERNLIBS}/Cg/bin" )
  ENDIF(EXISTS "$ENV{EXTERNLIBS}/cg")
ENDIF(CMAKE_CL_64) 

FIND_PROGRAM(CG_COMPILER cgc
  PATHS
  ${CG_BINPATH}
  $ENV{EXTERNLIBS}/cg/bin
  $ENV{EXTERNLIBS}/Cg/bin
  $ENV{PROGRAMFILES}/NVIDIA\ Corporation/Cg/bin
  $ENV{PROGRAMFILES}/Cg
  /usr/bin
  /usr/local/bin
  NO_DEFAULT_PATH
  DOC "NVIDIA Cg Compiler"
)

IF (APPLE)
  INCLUDE(${CMAKE_ROOT}/Modules/CMakeFindFrameworks.cmake)
  SET(CG_FRAMEWORK_INCLUDES)
  CMAKE_FIND_FRAMEWORKS(Cg)
  SET(Cg_FRAMEWORKS $ENV{EXTERNLIBS} ${Cg_FRAMEWORKS})
  SET(EXTERNLIBS $ENV{EXTERNLIBS})
  IF(Cg_FRAMEWORKS)
    FOREACH(dir ${Cg_FRAMEWORKS})
       #SET(CG_FRAMEWORK_INCLUDES ${CG_FRAMEWORK_INCLUDES} ${dir}/Headers ${dir}/PrivateHeaders)
       SET(CG_FRAMEWORK_INCLUDES ${CG_FRAMEWORK_INCLUDES} ${dir}/Cg.framework)
    ENDFOREACH(dir)
    # Find the include  dir
    FIND_PATH(CG_INCLUDE_DIR Headers/cg.h
      ${CG_FRAMEWORK_INCLUDES}
    )
    # Since we are using Cg framework, we must link to it.
    # Note, we use weak linking, so that it works even when Cg is not available.
    if (CG_INCLUDE_DIR)
       SET(CG_LIBRARY "-F${EXTERNLIBS} -framework Cg" CACHE STRING "Cg library")
       SET(CG_GL_LIBRARY "-F${EXTERNLIBS} -framework Cg" CACHE STRING "Cg GL library")
    endif(CG_INCLUDE_DIR)
  ENDIF(Cg_FRAMEWORKS)
  
ELSE (APPLE)

  IF(CG_COMPILER)
    GET_FILENAME_COMPONENT(CG_COMPILER_DIR "${CG_COMPILER}" PATH)
    GET_FILENAME_COMPONENT(CG_COMPILER_SUPER_DIR "${CG_COMPILER_DIR}" PATH)
  ELSE (CG_COMPILER)
    SET(CG_COMPILER_DIR .)
    SET(CG_COMPILER_SUPER_DIR ..)
  ENDIF(CG_COMPILER)

  FIND_PATH(CG_INCLUDE_DIR Cg/cg.h
    $ENV{EXTERNLIBS}/Cg/include
    $ENV{PROGRAMFILES}/NVIDIA\ Corporation/Cg/include
    $ENV{PROGRAMFILES}/Cg
    $ENV{CG_INC_PATH}
    /usr/include
    /usr/local/include
    ${CG_COMPILER_SUPER_DIR}/include
    ${CG_COMPILER_DIR}
	NO_DEFAULT_PATH
    DOC "The directory where Cg/cg.h resides"
  )

  FIND_LIBRARY(CG_LIBRARY
    NAMES Cg
    PATHS
	${CG_LIBPATH}
    $ENV{CG_LIB_PATH}
    $ENV{PROGRAMFILES}/NVIDIA\ Corporation/Cg/lib
    $ENV{PROGRAMFILES}/Cg
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
    ${CG_COMPILER_SUPER_DIR}/lib64
    ${CG_COMPILER_SUPER_DIR}/lib
    ${CG_COMPILER_DIR}
	NO_DEFAULT_PATH
    DOC "The Cg runtime library"
  )

  FIND_LIBRARY(CG_GL_LIBRARY
    NAMES CgGL
    PATHS
	${CG_LIBPATH}
    $ENV{CG_LIB_PATH}
    $ENV{PROGRAMFILES}/NVIDIA\ Corporation/Cg/lib
    $ENV{PROGRAMFILES}/Cg
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
    ${CG_COMPILER_SUPER_DIR}/lib64
    ${CG_COMPILER_SUPER_DIR}/lib
    ${CG_COMPILER_DIR}
	NO_DEFAULT_PATH
    DOC "The Cg runtime library"
  )
    
ENDIF (APPLE)

IF(CG_INCLUDE_DIR)
  SET(CG_INCLUDE_DIRS "${CG_INCLUDE_DIR}")
ENDIF(CG_INCLUDE_DIR)

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Cg DEFAULT_MSG CG_LIBRARY CG_GL_LIBRARY CG_INCLUDE_DIR CG_COMPILER)
MARK_AS_ADVANCED(CG_LIBRARY CG_GL_LIBRARY CG_INCLUDE_DIR CG_COMPILER)
