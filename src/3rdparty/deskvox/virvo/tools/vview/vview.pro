!include($$(COFRAMEWORKDIR)/mkspecs/config-first.pri):error(include of config-first.pri failed)
### don't modify anything before this line ###

TARGET		   = vview

PROJECT        = General

TEMPLATE       = app

CONFIG		   += console virvo glut
CUDA = $$(CUDA_DEFINES)
contains(CUDA,HAVE_CUDA) {
        DEFINES *= HAVE_CUDA
}

SOURCES        = vvview.cpp vvperformancetest.cpp vvobjview.cpp
HEADERS        = vvview.h vvperformancetest.h vvobjview.h


### don't modify anything below this line ###
!include ($$(COFRAMEWORKDIR)/mkspecs/config-last.pri):error(include of config-last.pri failed)
