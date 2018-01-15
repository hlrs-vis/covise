# - Find EIGEN
# Find the EIGEN includes and library
#
#  EIGEN_INCLUDE_DIR - Where to find EIGEN includes
#  EIGEN_LIBRARIES   - List of libraries when using EIGEN
#  EIGEN_FOUND       - True if EIGEN was found

IF(EIGEN_INCLUDE_DIR)
  SET(EIGEN_FIND_QUIETLY TRUE)
ENDIF(EIGEN_INCLUDE_DIR)

FIND_PATH(EIGEN_INCLUDE_DIR "Eigen/Eigen"
  PATHS
  $ENV{EIGEN_HOME}/include
  $ENV{EXTERNLIBS}/eigen3/include
  $ENV{EXTERNLIBS}/eigen/include
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw # Fink
  /opt/local # DarwinPorts
  /opt/csw # Blastwave
  /opt
  DOC "EIGEN - Headers"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
 

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(EIGEN DEFAULT_MSG EIGEN_INCLUDE_DIR)

  MARK_AS_ADVANCED( EIGEN_INCLUDE_DIR)
  
ELSE(MSVC)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(EIGEN DEFAULT_MSG EIGEN_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(EIGEN_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(EIGEN_FOUND)
  SET(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIR})
ENDIF(EIGEN_FOUND)