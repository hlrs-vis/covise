cmake_minimum_required (VERSION 2.6)
project (VRMLEXP)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmake;${CMAKE_MODULE_PATH}")

IF(WIN32)
  SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmake/windows;${CMAKE_MODULE_PATH}")
ENDIF()

FIND_PACKAGE(3DSMAX)

INCLUDE_DIRECTORIES(${3DSMAX_INCLUDE_DIRS})
INCLUDE_DIRECTORIES($ENV{DXSDK_DIR}/include)

INCLUDE(CoviseHelperMacros)

FIND_PACKAGE(CAL3D)
IF (NOT CAL3D_FOUND)
    MESSAGE("Skipping because of missing CAL3D")
    RETURN()
ENDIF(NOT CAL3D_FOUND)
INCLUDE_DIRECTORIES(${CAL3D_INCLUDE_DIR})
SET(EXTRA_LIBS ${EXTRA_LIBS} ${CAL3D_LIBRARIES})
LINK_DIRECTORIES(${3DSMAX_LIB_DIR})

INCLUDE_DIRECTORIES(${3DSMAX_INCLUDE_DIR})
SET(EXTRA_LIBS ${EXTRA_LIBS} ${3DSMAX_LIBRARIES})
INCLUDE_DIRECTORIES(/src/dxsdk/include)

INCLUDE_DIRECTORIES($ENV{COVISEDIR}/src/tools/vrmlexp)
INCLUDE_DIRECTORIES($ENV{COVISEDIR}/src/kernel/util)
ADD_DEFINITIONS( -DBYTESWAP)
ADD_DEFINITIONS( -D_STANDARD_C_PLUS_PLUS)
ADD_DEFINITIONS( -D_LEC_)
ADD_DEFINITIONS( -D_CRT_SECURE_NO_WARNINGS)
ADD_DEFINITIONS( -D_CRT_NON_CONFORMING_SWPRINTFS)

OPTION(COMPILE_UNICODE "Compile for unicode versions of max." ON)
IF(COMPILE_UNICODE)
ADD_DEFINITIONS( -D_UNICODE)
ADD_DEFINITIONS( -DUNICODE)
ENDIF(COMPILE_UNICODE)

SET(CMAKE_SHARED_LIBRARY_SUFFIX .dle)




SET(VRMLEXP_SOURCES 
anchor.cpp
arsensor.cpp
audio.cpp
backgrnd.cpp
bboard.cpp
bookmark.cpp
cal3dHelper.cpp
coTabletUI.cpp
cover.cpp
covise_connect.cpp
covise_conv.cpp
covise_host.cpp
covise_msg.cpp
covise_socket.cpp
cppout.cpp
defuse.cpp
dllmain.cpp
evalcol.cpp
fog.cpp
inline.cpp
COVISEObject.cpp
lod.cpp
MultiTouchSensor.cpp
navinfo.cpp
onoff.cpp
pmesh.cpp
polycnt.cpp
prox.cpp
resettm.cpp
script.cpp
Sky.cpp
sound.cpp
switch.cpp
tabletui.cpp
timer.cpp
touch.cpp
TUIParam.cpp
uselod.cpp
vrml2.cpp
vrml_api.cpp
vrmlexp.cpp
vrmlpch.cpp
BufferedStream.cpp
)
SET(VRMLEXP_HEADERS
anchor.h
appd.h
arsensor.h
audio.h
backgrnd.h
bboard.h
bookmark.h
cal3dHelper.h
coDLList.h
coTabletUI.h
coTypes.h
cover.h
covise_byteswap.h
covise_connect.h
covise_conv.h
covise_host.h
covise_list.h
covise_msg.h
covise_socket.h
defuse.h
evalcol.h
fog.h
inline.h
COVISEObject.h
inlist.h
lodctrl.h
lod.h
MultiTouchSensor.h
navinfo.h
normtab.h
ntrack.h
onoff.h
pmesh.h
prox.h
resource.h
script.h
Sky.h
sound.h
switch.h
tabletui.h
target.h
timer.h
touch.h
TUIParam.h
veroverrides.h
vrml2.h
vrml_api.h
vrmlexp.h
vrml.h
BufferedStream.h
)
SET(VRMLEXP_RESOURCES 
vrmlexp.rc
lodcurs.cur
animcurs.cur
crosshr1.cur
dmtlbut.bmp
dmtlmsk.bmp
vrmlexp.def
)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}) 


add_library (Vrmlexp SHARED ${VRMLEXP_SOURCES} ${VRMLEXP_HEADERS} ${VRMLEXP_RESOURCES})
target_link_libraries (Vrmlexp ${EXTRA_LIBS} ) 
add_custom_command(TARGET Vrmlexp POST_BUILD 
  COMMAND "${CMAKE_COMMAND}" -E copy 
     "$<TARGET_FILE:Vrmlexp>"
     ${3DSMAX_STDPLUGS_DIR} 
  COMMENT "Copying lib to output directory")
add_custom_command(TARGET Vrmlexp POST_BUILD debug
  COMMAND "${CMAKE_COMMAND}" -E copy 
     "$<TARGET_FILE_DIR:Vrmlexp>/vrmlexp.pdb"
     ${3DSMAX_STDPLUGS_DIR} 
  COMMENT "Copying pdb to output directory")

