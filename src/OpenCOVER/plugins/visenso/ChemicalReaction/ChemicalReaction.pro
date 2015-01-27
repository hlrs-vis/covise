!include($$(COFRAMEWORKDIR)/mkspecs/config-first.pri):error(include of config-first.pri failed)
### don't modify anything before this line ###

TARGET		= ChemicalReaction
PROJECT     = visenso

TEMPLATE    = opencoverplugin

CONFIG          *= boost vtrans grmsg openpluginutil

SOURCES     = \
        ChemicalReactionPlugin.cpp \
        Atom.cpp \
        Molecule.cpp \
        StartMolecule.cpp \
        EndMolecule.cpp \
        Design.cpp \
        DesignLib.cpp \
        MoleculeHandler.cpp \
        ReactionArea.cpp \
        StartButton.cpp \
        Equation.cpp



EXTRASOURCES    = \
        *.h


### don't modify anything below this line ###
!include ($$(COFRAMEWORKDIR)/mkspecs/config-last.pri):error(include of config-last.pri failed)
