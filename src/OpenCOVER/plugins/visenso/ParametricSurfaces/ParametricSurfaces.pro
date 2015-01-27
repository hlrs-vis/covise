!include($$(COFRAMEWORKDIR)/mkspecs/config-first.pri):error(include of config-first.pri failed)
### don't modify anything before this line ###

TARGET      = ParametricSurfaces
PROJECT     = General

TEMPLATE    = opencoverplugin

CONFIG     *= boost vtrans grmsg openpluginutil

LIBS        *= -lcoImage -lvtrans

SOURCES     = \
        ParametricSurfaces.cpp \
        algebra.cpp \
        diff.cpp \
        EPRINT.CPP \
        EVAL.CPP \
        HfT_osg_FindNode.cpp \
        HfT_osg_Plugin01_Animation.cpp \
        HfT_osg_Plugin01_Cons.cpp \
        HfT_osg_Plugin01_NormalschnittAnimation.cpp \
        HfT_osg_Plugin01_ParametricPlane.cpp \
        HfT_osg_Plugin01_ParametricSurface.cpp \
        HfT_osg_Plugin01_ReadSurfDescription.cpp \
        HfT_osg_ReadTextureImage.cpp \
        HfT_osg_StateSet.cpp \
        HfT_string.cpp \
        HlCAS.cpp \
        numeric.cpp \
       	PARSER.CPP \
        scanner.cpp \
        simp.cpp 

INCLUDEPATH    +=  /work/ac_te/INCLUDES

EXTRASOURCES    = *.h

### don't modify anything below this line ###
!include ($$(COFRAMEWORKDIR)/mkspecs/config-last.pri):error(include of config-last.pri failed)
