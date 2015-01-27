##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Uwe Woessner, Andreas Wierse  Date: 29.08.97 ##
##########################################################

                          # name of the module executable
APPLICATION 	= ReadVDAFS
                          # name of the module category
CATEGORY	= IO
                          # needed for creating distribution tar-files
PROJECT		= General

# put all object files that are needed to build the module here:

OBJECTS         = $(OBJDIR)/Name.o $(OBJDIR)/parser.o $(OBJDIR)/VDAFS.o \
                  $(OBJDIR)/nurbs.o


# $(APPLICATION) is also the name of the main source and include file!!!)

##########################################################

# modify this for your own path if necessary
COVISEDIR = ../../../..

# All common UNIX compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:


# All common rules are in this central makefile.

include $(COVISEDIR)/src/make.rules
#CC = insure

############### link rule for application module #########
# can be modified (e.g. for dependencies or libraries)
##########################################################

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJDIR)/$(APPLICATION).o $(OBJECTS)
	  $(CC) $(CPPFLAGS) -L/vis_swt/leda/LEDA-341 -o $@ $(OBJECTS) $(OBJDIR)/$(APPLICATION).o \
	  $(APPLLIB) $(LDLIBS) -lm -lL

############### general object rule ######################
# can be modified (e.g. for dependencies)
##########################################################

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp
	$(CC) $(CPPFLAGS) -I/vis_swt/leda/LEDA-341/incl -c $(APPLICATION).cpp \
							-o $(OBJDIR)/$(APPLICATION).o
	$(CC) $(CPPFLAGS) -I/vis_swt/leda/LEDA-341/incl -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	$(POSTCC)

############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################

$(OBJDIR)/Name.o: Name.cpp
	$(CC) $(CPPFLAGS) -I/vis_swt/leda/LEDA-341/incl  -c Name.cpp -o $(OBJDIR)/Name.o
	$(POSTCC)

$(OBJDIR)/parser.o: parser.cpp Name.h
	$(CC) $(CPPFLAGS) -I/vis_swt/leda/LEDA-341/incl -c parser.cpp \
                                                        -o $(OBJDIR)/parser.o 
        $(POSTCC)

$(OBJDIR)/VDAFS.o: VDAFS.cpp VDAFS.h Vec.h Matrix.h
	$(CC) $(CPPFLAGS) -I/vis_swt/leda/LEDA-341/incl -c VDAFS.cpp \
	                                                -o $(OBJDIR)/VDAFS.o $(POSTCC)
	$(POSTCC)

$(OBJDIR)/nurbs.o: nurbs.cpp Vec.h 
	$(CC) $(CPPFLAGS) -I/vis_swt/leda/LEDA-341/incl -c nurbs.cpp \
                                                        -o $(OBJDIR)/nurbs.o $(POSTCC)

	$(POSTCC)
