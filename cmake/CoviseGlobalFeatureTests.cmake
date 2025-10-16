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
   include(CheckCXXCompilerFlag)
   check_cxx_compiler_flag("-Wno-stdlibcxx-not-found" have_wno_stdlibcxx_not_found)
   if(${have_wno_stdlibcxx_not_found})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stdlibcxx-not-found")
   endif()
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-command-line-argument")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif()

IF(COVISE_USE_VISIBILITY)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)
ENDIF()
  
