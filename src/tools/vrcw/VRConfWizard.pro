TEMPLATE = app
TARGET = VRConfWizard
QT += core \
    gui \
    widgets \
    network \
    xml
HEADERS += vrcwstart.h \
    vrcwperson.h \
    vrcwsensortracksysdim.h \
    vrcwhsensbindex.h \
    vrcwutils.h \
    datatypes.h \
    vrcwprojectionressizetiled.h \
    vrcwprojectionvposfloor.h \
    xmlconfigwriter.h \
    vrcwfinal.h \
    tablemodel.h \
    vrcwhostprojection.h \
    vrcwprojectionressize.h \
    vrcwprojectiondimcave.h \
    vrcwtrackingdim.h \
    vrcwtrackinghw.h \
    vrcwprojectiondimpowerwall.h \
    vrcwprojectionhw.h \
    vrcwbase.h \
    vrcwtemplate.h \
    vrcwhost.h \
    vrconfwizard.h
SOURCES += vrcwstart.cpp \
    vrcwperson.cpp \
    vrcwsensortracksysdim.cpp \
    vrcwhsensbindex.cpp \
    vrcwutils.cpp \
    datatypes.cpp \
    vrcwprojectionressizetiled.cpp \
    vrcwprojectionvposfloor.cpp \
    xmlconfigwriter.cpp \
    vrcwfinal.cpp \
    tablemodel.cpp \
    vrcwhostprojection.cpp \
    vrcwprojectionressize.cpp \
    vrcwprojectiondimcave.cpp \
    vrcwtrackingdim.cpp \
    vrcwtrackinghw.cpp \
    vrcwprojectiondimpowerwall.cpp \
    vrcwprojectionhw.cpp \
    vrcwbase.cpp \
    vrcwtemplate.cpp \
    vrcwhost.cpp \
    main.cpp \
    vrconfwizard.cpp
FORMS += vrcwstart.ui \
    vrcwperson.ui \
    vrcwsensortracksysdim.ui \
    vrcwhsensbindex.ui \
    vrcwprojectionressizetiled.ui \
    vrcwprojectionvposfloor.ui \
    vrcwfinal.ui \
    vrcwhostprojection.ui \
    vrcwprojectionressize.ui \
    vrcwprojectiondimcave.ui \
    vrcwtrackingdim.ui \
    vrcwtrackinghw.ui \
    vrcwprojectiondimpowerwall.ui \
    vrcwprojectionhw.ui \
    vrcwtemplate.ui \
    vrcwhost.ui \
    vrconfwizard.ui
RESOURCES +=
CONFIG += console
