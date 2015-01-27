##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Uwe Woessner, Andreas Wierse  Date: 29.08.97 ##
##########################################################

                          # name of the module executable
APPLICATION 	= Tracer
                          # name of the module category
CATEGORY	= Tracer
                          # needed for creating distribution tar-files
PROJECT		= General

# put all object files that are needed to build the module here:

OBJECTS         = $(OBJDIR)/slinm.o $(OBJDIR)/slinb.o $(OBJDIR)/lblc.o $(OBJDIR)/x2xm.o $(OBJDIR)/rae.o \
	$(OBJDIR)/intevb.o 


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

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJDIR)/$(APPLICATION).o $(OBJECTS)
	  $(CC) $(CPPFLAGS) -o $@ $(OBJECTS) $(OBJDIR)/$(APPLICATION).o \
	  $(APPLLIB) $(LDLIBS) $(LIBF77) -lm

############### general object rule ######################
# can be modified (e.g. for dependencies)
##########################################################

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp $(APPLICATION).h
	$(CC) $(CPPFLAGS) -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	$(POSTCC)

############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################


$(OBJDIR)/slinm.o: slinm.f
	$(FF) $(FFLAGS) -c slinm.f -o $(OBJDIR)/slinm.o

$(OBJDIR)/slinb.o: slinb.f
	$(FF) $(FFLAGS) -c slinb.f -o $(OBJDIR)/slinb.o

$(OBJDIR)/lblc.o: lblc.f
	$(FF) $(FFLAGS) -c lblc.f -o $(OBJDIR)/lblc.o

$(OBJDIR)/x2xm.o: x2xm.f
	$(FF) $(FFLAGS) -c x2xm.f -o $(OBJDIR)/x2xm.o

$(OBJDIR)/rae.o: rae.f
	$(FF) $(FFLAGS) -c rae.f -o $(OBJDIR)/rae.o

$(OBJDIR)/intevb.o: intevb.f
	$(FF) $(FFLAGS) -c intevb.f -o $(OBJDIR)/intevb.o
