MACRO(USE_LAMURE)
    COVISE_FIND_PACKAGE(LAMURE)
    IF ((NOT LAMURE_FOUND)  AND (${ARGC} LESS 1))
        USING_MESSAGE("Skipping because of missing Lamure")
        RETURN()
    ENDIF((NOT LAMURE_FOUND) AND (${ARGC} LESS 1))
    COVISE_FIND_PACKAGE(SCHISM)
    COVISE_FIND_PACKAGE(FREEIMAGE)
    IF(NOT LAMURE_USED AND LAMURE_FOUND)
        SET(LAMURE_USED TRUE)
        INCLUDE_DIRECTORIES(SYSTEM ${LAMURE_INCLUDE_DIR} ${SCHISM_INCLUDE_DIRS})
        SET(EXTRA_LIBS ${EXTRA_LIBS} ${LAMURE_LIBRARIES})
        SET(EXTRA_LIBS ${EXTRA_LIBS} ${SCHISM_LIBRARIES})
    ENDIF()
ENDMACRO(USE_LAMURE)

