##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Uwe Woessner, Andreas Wierse  Date: 29.08.97 ##
##########################################################

                          # name of the module executable
APPLICATION = ReadAramcoSim
                          # name of the module category
CATEGORY    = Aramco
                          # needed for creating distribution tar-files
PROJECT     = Aramco


# $(APPLICATION) is also the name of the main source and include file!!!)

##########################################################

# All system-dependent compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:
OBJECTS = $(OBJDIR)/AramcoSimFile.o

# All common rules are in this central makefile.

include $(COVISEDIR)/src/make.rules

############### link rule for application module #########
# can be modified (e.g. for dependencies or libraries)
##########################################################

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJECTS) $(OBJDIR)/$(APPLICATION).o ext2SV.h
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
$(OBJDIR)/AramcoSimFile.o: AramcoSimFile.cpp AramcoSimFile.h
	$(CC) $(CPPFLAGS) -c AramcoSimFile.cpp -o $(OBJDIR)/AramcoSimFile.o
	$(POSTCC)


