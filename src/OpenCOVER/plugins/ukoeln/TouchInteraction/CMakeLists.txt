# PROJECT(TouchInteraction)

COVISE_FIND_PACKAGE(BLAS)
COVISE_FIND_PACKAGE(LAPACK)

IF(NOT BLAS_FOUND OR NOT LAPACK_FOUND)
    RETURN()
ENDIF()

INCLUDE_DIRECTORIES(  
    levmar
)

SET(HEADERS
    Plugin.h
    ScreenSpaceHandler.h
    TapHandler.h
    Touch.h
    TouchHandler.h
    TouchInteraction.h
    Utouch3DHandler.h
    levmar/levmar.h
)

SET(SOURCES
    Plugin.cpp
    ScreenSpaceHandler.cpp
    TapHandler.cpp
    TouchInteraction.cpp
    Utouch3DHandler.cpp
    levmar/Axb.c
    levmar/lm.c
    levmar/lmbc.c
    levmar/lmblec.c
    levmar/lmbleic.c
    levmar/lmlec.c
    levmar/misc.c
)

cover_add_plugin(TouchInteraction)

TARGET_LINK_LIBRARIES(TouchInteraction ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
