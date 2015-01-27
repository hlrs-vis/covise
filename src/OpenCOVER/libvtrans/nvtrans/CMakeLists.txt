SET(Boost_USE_STATIC_LIBS   ON)
SET(Boost_USE_MULTITHREADED ON)

OPTION (BUILD_SHARED_NVTRANS_LIBS "Build shared nvtrans lib." ON)

FIND_PACKAGE(Boost COMPONENTS locale thread system REQUIRED)
IF(Boost_system_FOUND)
ADD_DEFINITIONS(-DBOOST_SYSTEM_FOUND)
ENDIF(Boost_system_FOUND)

ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
REMOVE_DEFINITIONS(-D_GTEST_ENABLED)

INCLUDE_DIRECTORIES(${COMMON_INCLUDES} ${Boost_INCLUDE_DIRS} "../vtrans/")
#thread date_time 
#SET(Boost_ADDITIONAL_VERSIONS "1.44" "1.44.0")
#ADD_DEFINITIONS(-DBOOST_THREAD_USE_LIB)

#${LINK_DIRECTORIES} 
#CMAKE_POLICY(SET CMP0015 NEW)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

OPTION(
  gtest_force_shared_crt
  "Use shared (DLL) run-time lib even when Google Test is built as static lib."
  ON)

SET(LIB_TYPE STATIC)
IF (BUILD_SHARED_NVTRANS_LIBS)
    SET (LIB_TYPE SHARED)
ENDIF (BUILD_SHARED_NVTRANS_LIBS)

ADD_LIBRARY(nvtrans ${LIB_TYPE} NvTranslator.h 
                                NvTranslator.cpp
                                ../vtrans/PathTranslator.cpp
                                ../vtrans/PathTranslator.h
                                ../vtrans/vtrans.h
                                ../vtrans/vtrans.cpp 
                                ../vtrans/TransFactory.h 
                                ../vtrans/TransFactory.cpp 
                                ../vtrans/BoostTranslator.h 
                                ../vtrans/BoostTranslator.cpp 
                                ../vtrans/ITranslator.h 
                                ../vtrans/PathTranslator.h 
                                ../vtrans/PathTranslator.cpp)
                                
TARGET_LINK_LIBRARIES(nvtrans  ${Boost_LIBRARIES})

IF(CMAKE_CXX_FLAGS_DEBUG MATCHES "/RTC1")
    STRING(REPLACE "/RTC1"  "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
ENDIF()

IF(CMAKE_CXX_FLAGS MATCHES "/EHsc")
    STRING(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ENDIF()

SET_TARGET_PROPERTIES(nvtrans PROPERTIES COMPILE_FLAGS "/clr")
#SET_TARGET_PROPERTIES(nvtrans PROPERTIES COMPILE_FLAGS "/clr") # /EHa
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clr")

IF ( MSVC60 OR MSVC70 OR MSVC71 OR MSVC80 OR MSVC90 )
    MESSAGE(AUTHOR_WARNING "\nDue some problems with Visual Studio Express 2008\nyou have to enable the Common Runtime Language Support\nfor this project MANUALLY!\nOr build this library with at least VS 2010!\n")
ENDIF ( MSVC60 OR MSVC70 OR MSVC71 OR MSVC80 OR MSVC90 )

                                
INSTALL(TARGETS nvtrans
        DESTINATION lib)