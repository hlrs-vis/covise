#!smake
 ##########################################################################
##                                                           (C)1995 RUS  ##
##                                                                        ##
## Description: Makefile for COVISE modules                               ##
##                                                                        ##
##                                                                        ##
##                                                                        ##
##                                                                        ##
##                                                                        ##
##                                                                        ##
##                               (C) 1995                                 ##
##                             Uwe Woessner                               ##
##                Computer Center University of Stuttgart                 ##
##                            Allmandring 30                              ##
##                            70550 Stuttgart                             ##  
## Author: Uwe Woessner                                                   ##
##                                                                        ##
## Date:  09.08.96  V1.0                                                  ##
 ##########################################################################

# Replace this with your own path if necessary
#COVISEDIR = ../../../..

# All common UNIX compiler flags and options are now in
# this central makefile.

#include $(ROOT)/usr/include/make/commondefs
#include $(OPROOT)/usr/share/Optimizer/src/opusercommondefs 
COVISEDIR = ../../../..

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

#
# Tolerant flag indicates that the module will
# not throw an error when data at the input port
# is missing for execution. 
TOLERANT = -DTOLERANT
#TOLERANT = 
#
# Performance measurement stuff
# -----------------------------
#
# set the flag for timing measurements
# A protocol file called 'Application.time'
# will be written out after a session
#
TIMEFLAG = 
#TIMEFLAG  = -DTIMER

PRECSUF  = dp

CPPFLAGS2 = -DOP_REAL_IS_DOUBLE -D_SGI_SOURCE -DOPENGL -DMDG -DCSNORMALSET3S -DCOMPILE_ACTION -DOPENGL -DOPENGL_EXTENSIONS -g 

#LFLAGS2 = -L/disk2/people/mwierse/HOME/optimizer/src/apps/optimizeDemo -lop_$(PRECSUF) -lGL
LFLAGS2 =  -lop_$(PRECSUF) -lGL -lcosmo3d

#################################################################################
# Application module stuff
#################################################################################
APPLICATION 	= OptimizerReducer
CATEGORY	= Filter
PROJECT		= General

OBJECTS         = $(OBJDIR)/ApplMain.o \
                 $(OBJDIR)/HandleSet.o  

#OBJECTS         = $(OBJDIR)/ApplMain.o \
#                  $(OBJDIR)/HandleSet.o
#

all:	$(OBJDIR) $(PLATFORM)/bin/$(CATEGORY) $(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION)

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJDIR)/$(APPLICATION).o $(OBJECTS)
	  $(CC) $(CPPFLAGS) $(CPPFLAGS2) -o $@ $(OBJDIR)/$(APPLICATION).o $(OBJECTS) \
	  $(LDLIBS) $(LFLAGS) $(LFLAGS2) $(APPLLIB) -lm

$(PLATFORM)/bin/$(CATEGORY):
	mkdir $(PLATFORM)/bin/$(CATEGORY)

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp $(APPLICATION).h
	$(CC) $(CPPFLAGS) $(CPPFLAGS2) -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	$(POSTCC)

$(OBJDIR)/ApplMain.o: ApplMain.cpp $(APPLICATION).h
	$(CC) $(CPPFLAGS) $(CPPFLAGS2) -c ApplMain.cpp -o $(OBJDIR)/ApplMain.o
	$(POSTCC)

$(OBJDIR)/HandleSet.o: HandleSet.cpp $(APPLICATION).h
	$(CC) $(CPPFLAGS) $(CPPFLAGS2) -c HandleSet.cpp -o $(OBJDIR)/HandleSet.o
	$(POSTCC)
	
$(OBJDIR):
	mkdir $(OBJDIR)	

clean: 
	-/bin/rm -rf $(OBJDIR) $(EXTRACLEAN)

THISDIR = src/application/general/OPTIMIZER
SRCFILES = $(THISDIR)/*.cpp $(THISDIR)/*.h $(THISDIR)/Makefile

tar_bin:
	cd $(COVISEDIR) ; \
	if (test -f covise_bin.$(PROJECT).$(ARCHSUFFIX).tar) \
	then tar uvf covise_bin.$(PROJECT).$(ARCHSUFFIX).tar $(ARCHSUFFIX)/bin/$(CATEGORY)/$(APPLICATION);\
	else tar cvf covise_bin.$(PROJECT).$(ARCHSUFFIX).tar $(ARCHSUFFIX)/bin/$(CATEGORY)/$(APPLICATION);\
	fi

tar_src:
	cd $(COVISEDIR) ; \
	if (test -f covise_src.$(PROJECT).$(ARCHSUFFIX).tar) \
	then tar uvf covise_src.$(PROJECT).$(ARCHSUFFIX).tar $(SRCFILES) $(LFLAGS) $(LFLAGS2) ;\
	else tar cvf covise_src.$(PROJECT).$(ARCHSUFFIX).tar $(SRCFILES) $(LFLAGS) $(LFLAGS2);\
	fi

# DO NOT DELETE

$(OBJDIR)/ApplMain.o: OptimizerReducer.h
$(OBJDIR)/HandleSet.o: OptimizerReducer.h
$(OBJDIR)/OptimizerReducer.o: OptimizerReducer.h 

