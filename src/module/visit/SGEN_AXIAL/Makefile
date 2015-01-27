##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Uwe Woessner, Andreas Wierse  Date: 29.08.97 ##
##########################################################

                          # name of the module executable
APPLICATION 	= SGen_fortran
                          # name of the module category
CATEGORY	= IO
                          # needed for creating distribution tar-files
PROJECT		= Ihs


# $(APPLICATION) is also the name of the main source and include file!!!)

OBJECTS       = $(OBJDIR)/geogen.o    \
		$(OBJDIR)/eingabe.o    \
		$(OBJDIR)/ausgabe.o    \
		$(OBJDIR)/rechnen.o    \
		$(OBJDIR)/profil.o     \
		$(OBJDIR)/axnet.o      \
		$(OBJDIR)/netzlesen.o 

##########################################################

# modify this for your own path if necessary
COVISEDIR = ../../../..

# All common UNIX compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)
#include $(COVISEDIR)/src/make.env.sgi

# if you want to modify settings of variables that are set in the
# central makefile, do this here:


# All common rules are in this central makefile.

include $(COVISEDIR)/src/make.rules

############### link rule for application module #########
# can be modified (e.g. for dependencies or libraries)
##########################################################

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJDIR)/$(APPLICATION).o $(OBJECTS)
	  $(CC) $(CPPFLAGS) -o $@ $(OBJECTS) $(OBJDIR)/$(APPLICATION).o \
	  $(LDLIBS) $(APPLLIB) -lftn  -lm

############### general object rule ######################
# can be modified (e.g. for dependencies)
##########################################################


############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp $(APPLICATION).h
	$(CC) $(CPPFLAGS) -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	$(POSTCC)

$(OBJDIR)/geogen.o: geogen.f
	$(FF) $(FFLAGS) -c geogen.f -o $(OBJDIR)/geogen.o

$(OBJDIR)/eingabe.o: eingabe.f
	$(FF) $(FFLAGS) -c eingabe.f -o $(OBJDIR)/eingabe.o

$(OBJDIR)/ausgabe.o: ausgabe.f
	$(FF) $(FFLAGS) -c ausgabe.f -o $(OBJDIR)/ausgabe.o

$(OBJDIR)/rechnen.o: rechnen.f
	$(FF) $(FFLAGS) -c rechnen.f -o $(OBJDIR)/rechnen.o

$(OBJDIR)/profil.o: profil.f
	$(FF) $(FFLAGS) -c profil.f -o $(OBJDIR)/profil.o

$(OBJDIR)/axnet.o: axnet.f
	$(FF) $(FFLAGS) -c axnet.f -o $(OBJDIR)/axnet.o

$(OBJDIR)/netzlesen.o: netzlesen.f
	$(FF) $(FFLAGS) -c netzlesen.f -o $(OBJDIR)/netzlesen.o




