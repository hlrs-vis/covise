# @file CoviseGlobalFeatureTests.cmake
#
# Place all the feature tests here you need in a global scope
#
# @author Blasius Czink


INCLUDE(TestBigEndian)
TEST_BIG_ENDIAN(COVISE_SYS_BIGENDIAN)
IF(COVISE_SYS_BIGENDIAN)
  # MESSAGE("BYTESWAP is OFF")
ELSE(COVISE_SYS_BIGENDIAN)
  # MESSAGE("BYTESWAP is ON")
  ADD_DEFINITIONS(-DBYTESWAP)
ENDIF(COVISE_SYS_BIGENDIAN)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -Wno-stdlibcxx-not-found")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-command-line-argument")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif()

IF(COVISE_USE_VISIBILITY)
IF(CMAKE_COMPILER_IS_GNUCXX)
   SET(COVISE_COMPILE_FLAGS "${COVISE_COMPILE_FLAGS} -fvisibility=hidden")
ENDIF()
IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   SET(COVISE_COMPILE_FLAGS "${COVISE_COMPILE_FLAGS} -fvisibility=hidden")
ENDIF()
ENDIF()
  
