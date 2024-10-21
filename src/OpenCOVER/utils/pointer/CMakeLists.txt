ADD_LIBRARY(coPointerUtil INTERFACE ResetOnCopy.h)
#additional stuff from covise_add_library which we can't use to build an interface library
SET_TARGET_PROPERTIES(coPointerUtil PROPERTIES OUTPUT_NAME "coPointerUtil")

IF("${MAIN_FOLDER}" STREQUAL "")
    set_target_properties(coPointerUtil PROPERTIES FOLDER "Libraries")
ELSE()
    set_target_properties(coPointerUtil PROPERTIES FOLDER "${MAIN_FOLDER}/Libraries")
ENDIF()
COVISE_ADJUST_OUTPUT_DIR(coPointerUtil)

# set additional COVISE_COMPILE_FLAGS
SET_TARGET_PROPERTIES(coPointerUtil PROPERTIES COMPILE_FLAGS "${COVISE_COMPILE_FLAGS}")
# set additional COVISE_LINK_FLAGS
SET_TARGET_PROPERTIES(coPointerUtil PROPERTIES LINK_FLAGS "${COVISE_LINK_FLAGS}")


COVISE_INSTALL_TARGET(coPointerUtil)
COVISE_INSTALL_HEADERS(coPointerUtil ResetOnCopy.h)
COVISE_WNOERROR(coPointerUtil BASEARCH zebu)
