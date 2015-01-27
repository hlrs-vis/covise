##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Andreas Werner                Date: 31.10.99 ##
##########################################################

                          # name of the module executable
APPLICATION 	= ReadCADMould
                          # name of the module category
CATEGORY	      = IO
                          # needed for creating distribution tar-files
PROJECT		   = General

# put all object files that are needed to build the module here:

OBJECTS         = $(OBJDIR)/CadmouldData.o 

##########################################################
# All system-dependent UNIX compiler flags and options are
# in this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:

SRCFILES = $${THISDIR}/*.cpp $${THISDIR}/*.h $${THISDIR}/Makefile
# All common rules are in this central makefile.

include $(COVISEDIR)/src/make.rules

############### link rule for application module #########
# can be modified (e.g. for dependencies or libraries)
##########################################################

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJDIR)/$(APPLICATION).o $(OBJECTS)
	  $(CC) $(CPPFLAGS) -o $@ $(OBJECTS) $(OBJDIR)/$(APPLICATION).o \
	        $(COVISE_LIBDIR) $(COVISEAPI) $(APPLLIB) $(COVISELIB)

############### general object rule ######################
# can be modified (e.g. for dependencies)
##########################################################

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp $(APPLICATION).h
	$(CC) $(CPPFLAGS) -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	$(POSTCC)

############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################
$(OBJDIR)/CadmouldData.o: CadmouldData.cpp
	$(CC) $(CPPFLAGS) -c CadmouldData.cpp -o $(OBJDIR)/CadmouldData.o
	$(POSTCC)

# DO NOT DELETE

$(OBJDIR)/CadmouldData.o: CadmouldData.h
$(OBJDIR)/ReadCADMould.o: ReadCADMould.h CadmouldData.h
