##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Uwe Woessner, Andreas Wierse  Date: 29.08.97 ##
##########################################################

                          # name of the module executable
APPLICATION     = ModifyAddPart
                          # name of the module category
CATEGORY        = Visit
                          # needed for creating distribution tar-files
PROJECT         = Visit

# put all object files that are needed to build the module here:

OBJECTS         =


# $(APPLICATION) is also the name of the main source and include file!!!)

##########################################################

# modify this for your own path if necessary
# COVISEDIR = ../../../..

# All common UNIX compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:


# All common rules are in this central makefile.

include $(COVISEDIR)/src/make.rules

############### link rule for application module #########
# can be modified (e.g. for dependencies or libraries)
##########################################################

ADD_INC = -I$(COVISEDIR)/src/application/visit/TETIN_LIB $(PERFORMER_INC_PATH)
$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJDIR)/$(APPLICATION).o $(OBJDIR)/domain.o $(OBJECTS)
	  $(CC) $(CPPFLAGS) -o $@ $(OBJECTS) $(OBJDIR)/$(APPLICATION).o \
		$(OBJDIR)/domain.o IcemProject_int.o \
	  domainlib.a -lTetin $(LDLIBS) $(PERFORMER_LIB) $(COVISEAPI) $(APPLLIB) \
		 -lftn -lm -lmalloc

############### general object rule ######################
# can be modified (e.g. for dependencies)

##########################################################
$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp
	$(LIBCC) $(APPLICATION).cpp $(LIBCPPFLAGS)
	$(POSTCC)

$(OBJDIR)/domain.o: domain.cpp
	$(LIBCC) domain.cpp $(LIBCPPFLAGS)
	$(POSTCC)



############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################
