!include($$(COFRAMEWORKDIR)/mkspecs/config-first.pri):error(include of config-first.pri failed)
### don't modify anything before this line ###

TARGET		= AlkaneBuilder
PROJECT     = visenso

TEMPLATE    = opencoverplugin

CONFIG      *= boost vtrans grmsg openpluginutil

SOURCES     = \
        AlkaneBuilderPlugin.cpp\
        AlkaneDatabase.cpp \
        AlkaneBuilder.cpp \
        Atom.cpp \
        Carbon.cpp \
        AtomBallInteractor.cpp \
        AtomStickInteractor.cpp \
        coVR3DRotCenterInteractor.cpp


HEADERS    = \
        AlkaneBuilderPlugin.h \
        AlkaneDatabase.h \
        AlkaneBuilder.h \
        Atom.h \
        Carbon.h \
        AtomBallInteractor.h \
        AtomStickInteractor.h \
        coVR3DRotCenterInteractor.h

### don't modify anything below this line ###
!include ($$(COFRAMEWORKDIR)/mkspecs/config-last.pri):error(include of config-last.pri failed)
