##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Andreas Werner              Date: 10.01.2000 ##
##########################################################

                          # name of the module executable
APPLICATION     = 3rdwave2cov
                          # name of the module category none - put in bin dir
CATEGORY        = 
                          # needed for creating distribution tar-files
PROJECT		= Develop

# put all object files that are needed to build the module here:

OBJECTS         = 

# $(APPLICATION) is also the name of the main source and include file!!!)

##########################################################

# All system-dependent UNIX compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:


# All common rules are in this central makefile.
include $(COVISEDIR)/src/make.rules

# Override binary distribution files
#BINFILES = $(ARCHSUFFIX)/bin/$(APPLICATION)


############### link rule for application module #########
# can be modified (e.g. for dependencies or libraries)
##########################################################

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJDIR)/$(APPLICATION).o $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $@ $(OBJECTS) $(OBJDIR)/$(APPLICATION).o 

############### general object rule ######################
# can be modified (e.g. for dependencies)
##########################################################

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp 
	$(CC) $(CPPFLAGS) $(COVISEFILELIBINC) -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	$(POSTCC)

############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################

$(OBJDIR)/CadmouldGrid.o: CadmouldGrid.cpp CadmouldGrid.h
	$(CC) $(CPPFLAGS) -c CadmouldGrid.cpp -o $(OBJDIR)/CadmouldGrid.o
	$(POSTCC)
