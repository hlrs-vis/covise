##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Uwe Woessner, Andreas Wierse  Date: 29.08.97 ##
##########################################################

                          # name of the module executable
APPLICATION 	= SPartToU
                          # name of the module category
CATEGORY	      = Tools
                          # needed for creating distribution tar-files
PROJECT		   = General

# put all object files that are needed to build the module here:

OBJECTS         = $(OBJDIR)/Map1D.o

##########################################################

# modify this for your own path if necessary
# COVISEDIR = ../../../..

# All common UNIX compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:
#OPT=-O3

SRCFILES = $${THISDIR}/*.cpp $${THISDIR}/*.h $${THISDIR}/Makefile
# All common rules are in this central makefile.

include $(COVISEDIR)/src/make.rules

# CC = insure -o32

############### link rule for application module #########
# can be modified (e.g. for dependencies or libraries)
##########################################################

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJDIR)/$(APPLICATION).o $(OBJECTS)
	  $(CC) $(CPPFLAGS) -o $@ $(OBJECTS) $(OBJDIR)/$(APPLICATION).o \
	        $(COVISE_LIBDIR) $(COVISEAPI) $(APPLLIB) \
	        $(COVISELIB) $(COVISEUTILLIB) -lm 

############### general object rule ######################
# can be modified (e.g. for dependencies)
##########################################################

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp $(APPLICATION).h 
	$(CC) $(CPPFLAGS) -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	$(POSTCC)

############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################

$(OBJDIR)/Map1D.o: Map1D.cpp Map1D.h
	$(CC) $(CPPFLAGS) -c Map1D.cpp -o $(OBJDIR)/Map1D.o
	$(POSTCC)
