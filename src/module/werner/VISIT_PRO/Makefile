##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Uwe Woessner, Andreas Wierse  Date: 29.08.97 ##
##########################################################

                          # name of the module executable
APPLICATION     = VisitPro
                          # name of the module category
CATEGORY        = DC
                          # needed for creating distribution tar-files
PROJECT         = DC_F1AK


# $(APPLICATION) is also the name of the main source and include file!!!)

##########################################################

# All system-dependent compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:

OBJECTS = $(OBJDIR)/Attachable.o \
          $(OBJDIR)/AddGeom.o \
          $(OBJDIR)/MultiFileParam.o \
          $(OBJDIR)/ReadObj.o

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

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp
	$(CC) $(CPPFLAGS) -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	$(POSTCC)

############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################

$(OBJDIR)/Attachable.o: Attachable.cpp
	$(CC) $(CPPFLAGS) -c Attachable.cpp -o $(OBJDIR)/Attachable.o
	$(POSTCC)

$(OBJDIR)/MultiFileParam.o: MultiFileParam.cpp
	$(CC) $(CPPFLAGS) -c MultiFileParam.cpp -o $(OBJDIR)/MultiFileParam.o
	$(POSTCC)

$(OBJDIR)/AddGeom.o: AddGeom.cpp
	$(CC) $(CPPFLAGS) -c AddGeom.cpp -o $(OBJDIR)/AddGeom.o
	$(POSTCC)

$(OBJDIR)/ReadObj.o: ReadObj.cpp
	$(CC) $(CPPFLAGS) -c ReadObj.cpp -o $(OBJDIR)/ReadObj.o
	$(POSTCC)

