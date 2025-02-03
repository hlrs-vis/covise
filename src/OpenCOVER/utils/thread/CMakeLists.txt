ADD_LIBRARY(coThreadUtil INTERFACE threadworker.h)
#additional stuff from covise_add_library which we can't use to build an interface library
SET_TARGET_PROPERTIES(coThreadUtil PROPERTIES OUTPUT_NAME "coThreadUtil")

IF("${MAIN_FOLDER}" STREQUAL "")
    set_target_properties(coThreadUtil PROPERTIES FOLDER "Libraries")
ELSE()
    set_target_properties(coThreadUtil PROPERTIES FOLDER "${MAIN_FOLDER}/Libraries")
ENDIF()
COVISE_ADJUST_OUTPUT_DIR(coThreadUtil)

# set additional COVISE_COMPILE_FLAGS
SET_TARGET_PROPERTIES(coThreadUtil PROPERTIES COMPILE_FLAGS "${COVISE_COMPILE_FLAGS}")
# set additional COVISE_LINK_FLAGS
SET_TARGET_PROPERTIES(coThreadUtil PROPERTIES LINK_FLAGS "${COVISE_LINK_FLAGS}")


COVISE_INSTALL_TARGET(coThreadUtil)
COVISE_INSTALL_HEADERS(coThreadUtil threadworker.h)
COVISE_WNOERROR(coThreadUtil BASEARCH zebu)
