##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Andreas Werner              Date: 10.01.2000 ##
##########################################################

                          # name of the module executable
APPLICATION     = magmaMesh2cov
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

all: $(PLATFORM)/bin/$(CATEGORY)/magmaData2cov
all: $(PLATFORM)/bin/$(CATEGORY)/magmaVect2cov 
all: $(PLATFORM)/bin/$(CATEGORY)/magmaFill2cov 

############### link rule for application module #########
# can be modified (e.g. for dependencies or libraries)
##########################################################

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJDIR)/$(APPLICATION).o $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $@ $(OBJECTS) $(OBJDIR)/$(APPLICATION).o 

$(PLATFORM)/bin/$(CATEGORY)/magmaData2cov: $(OBJDIR)/magmaData2cov.o $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $@ $(OBJECTS) $(OBJDIR)/magmaData2cov.o 

$(PLATFORM)/bin/$(CATEGORY)/magmaVect2cov: $(OBJDIR)/magmaVect2cov.o $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $@ $(OBJECTS) $(OBJDIR)/magmaVect2cov.o 

$(PLATFORM)/bin/$(CATEGORY)/magmaFill2cov: $(OBJDIR)/magmaFill2cov.o $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $@ $(OBJECTS) $(OBJDIR)/magmaFill2cov.o 

############### general object rule ######################
# can be modified (e.g. for dependencies)
##########################################################

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp 
	$(CC) $(CPPFLAGS) $(COVISEFILELIBINC) -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	$(POSTCC)

$(OBJDIR)/magmaData2cov.o: magmaData2cov.cpp 
	$(CC) $(CPPFLAGS) $(COVISEFILELIBINC) -c magmaData2cov.cpp -o $(OBJDIR)/magmaData2cov.o
	$(POSTCC)

$(OBJDIR)/magmaVect2cov.o: magmaVect2cov.cpp 
	$(CC) $(CPPFLAGS) $(COVISEFILELIBINC) -c magmaVect2cov.cpp -o $(OBJDIR)/magmaVect2cov.o
	$(POSTCC)

$(OBJDIR)/magmaFill2cov.o: magmaFill2cov.cpp 
	$(CC) $(CPPFLAGS) $(COVISEFILELIBINC) -c magmaFill2cov.cpp -o $(OBJDIR)/magmaFill2cov.o
	$(POSTCC)

############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################

$(OBJDIR)/CadmouldGrid.o: CadmouldGrid.cpp CadmouldGrid.h
	$(CC) $(CPPFLAGS) -c CadmouldGrid.cpp -o $(OBJDIR)/CadmouldGrid.o
	$(POSTCC)
