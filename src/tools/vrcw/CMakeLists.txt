if (COVISE_USE_QT4)
    return()
endif()

# PROJECT(VRConfWizard)

set(HEADERS
datatypes.h tablemodel.h vrconfwizard.h vrcwbase.h vrcwfinal.h vrcwhost.h vrcwhostprojection.h vrcwhsensbindex.h vrcwperson.h vrcwprojectiondimcave.h vrcwprojectiondimpowerwall.h vrcwprojectionhw.h vrcwprojectionressize.h vrcwprojectionressizetiled.h vrcwprojectionvposfloor.h vrcwsensortracksysdim.h vrcwstart.h vrcwtemplate.h vrcwtrackingdim.h vrcwtrackinghw.h vrcwutils.h xmlconfigwriter.h
)

qt_wrap_cpp(MOC_SOURCES
tablemodel.h vrconfwizard.h vrcwfinal.h vrcwhost.h vrcwhostprojection.h vrcwhsensbindex.h vrcwperson.h vrcwprojectiondimcave.h vrcwprojectiondimpowerwall.h vrcwprojectionhw.h vrcwprojectionressize.h vrcwprojectionressizetiled.h vrcwprojectionvposfloor.h vrcwsensortracksysdim.h vrcwstart.h vrcwtemplate.h vrcwtrackingdim.h vrcwtrackinghw.h
)

set(SOURCES
datatypes.cpp main.cpp tablemodel.cpp vrconfwizard.cpp vrcwbase.cpp vrcwfinal.cpp vrcwhost.cpp vrcwhostprojection.cpp vrcwhsensbindex.cpp vrcwperson.cpp vrcwprojectiondimcave.cpp vrcwprojectiondimpowerwall.cpp vrcwprojectionhw.cpp vrcwprojectionressize.cpp vrcwprojectionressizetiled.cpp vrcwprojectionvposfloor.cpp vrcwsensortracksysdim.cpp vrcwstart.cpp vrcwtemplate.cpp vrcwtrackingdim.cpp vrcwtrackinghw.cpp vrcwutils.cpp xmlconfigwriter.cpp
)

qt_wrap_ui(UI_SOURCES
vrconfwizard.ui vrcwfinal.ui vrcwhost.ui vrcwhostprojection.ui vrcwhsensbindex.ui vrcwperson.ui vrcwprojectiondimcave.ui vrcwprojectiondimpowerwall.ui vrcwprojectionhw.ui vrcwprojectionressize.ui vrcwprojectionressizetiled.ui vrcwprojectionvposfloor.ui vrcwsensortracksysdim.ui vrcwstart.ui vrcwtemplate.ui vrcwtrackingdim.ui vrcwtrackinghw.ui
)

qt_add_resources(RESOURCES
)

add_covise_executable(VRConfWizard MACOSX_BUNDLE ${SOURCES} ${HEADERS} ${UI_SOURCES} ${RESOURCES} ${MOC_SOURCES})
qt_use_modules(VRConfWizard OpenGL Widgets Network Xml)

covise_install_target(VRConfWizard)

target_link_libraries(VRConfWizard
)
