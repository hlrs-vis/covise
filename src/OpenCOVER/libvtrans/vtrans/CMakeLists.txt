SET(Boost_USE_STATIC_LIBS   OFF)
SET(Boost_USE_MULTITHREADED ON)

OPTION (BUILD_SHARED_VTRANS_LIBS "Build shared vtrans lib." OFF)

FIND_PACKAGE(Boost COMPONENTS locale thread system REQUIRED)

#FIND_PACKAGE(Boost COMPONENTS system OPTIONAL)
IF(Boost_system_FOUND)
ADD_DEFINITIONS(-DBOOST_SYSTEM_FOUND)
ENDIF(Boost_system_FOUND)

INCLUDE_DIRECTORIES(${COMMON_INCLUDES} ${Boost_INCLUDE_DIRS})
#thread date_time 
#SET(Boost_ADDITIONAL_VERSIONS "1.44" "1.44.0")
#ADD_DEFINITIONS(-DBOOST_THREAD_USE_LIB)

#${LINK_DIRECTORIES} 
#CMAKE_POLICY(SET CMP0015 NEW)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

IF (BUILD_UNIT_TESTS)
    INCLUDE_DIRECTORIES(${COMMON_INCLUDES} ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
ENDIF(BUILD_UNIT_TESTS)

#SET(gtest_force_shared_crt ON CACHE BOOL "Don't force shared." FORCE)
OPTION(
  gtest_force_shared_crt
  "Use shared (DLL) run-time lib even when Google Test is built as static lib."
  ON)

SET(LIB_TYPE STATIC)
IF (BUILD_SHARED_VTRANS_LIBS)
    SET (LIB_TYPE SHARED)
ENDIF (BUILD_SHARED_VTRANS_LIBS)

#IF(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_COMPILER_IS_GNUCC)
IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ENDIF()

ADD_COVISE_LIBRARY(vtrans ${LIB_TYPE} vtrans.h vtrans.cpp TransFactory.h TransFactory.cpp BoostTranslator.h BoostTranslator.cpp ITranslator.h PathTranslator.h PathTranslator.cpp vtransexport.h vtransexport.cpp)
if(NOT MSVC)
    add_covise_compile_flags(vtrans "-Wno-error=deprecated-declarations")
endif()
TARGET_LINK_LIBRARIES(vtrans  ${Boost_LIBRARIES})

FIND_PACKAGE(ICU)
IF (ICU_FOUND)
    INCLUDE_DIRECTORIES(${COMMON_INCLUDES} ${ICU_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(vtrans  ${ICU_LIBRARIES} ${ICU_I18N_LIBRARIES})
    FIND_PACKAGE (Threads)
    TARGET_LINK_LIBRARIES (vtrans ${CMAKE_THREAD_LIBS_INIT})
ENDIF(ICU_FOUND)

IF (NOT COVISE_BUILD)
    INSTALL(FILES ITranslator.h TransFactory.h vtrans.h PathTranslator.h
            DESTINATION include)

    INSTALL(TARGETS vtrans
            DESTINATION lib)
ENDIF()

COVISE_INSTALL_TARGET(vtrans)
