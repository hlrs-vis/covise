##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Uwe Woessner, Andreas Wierse  Date: 29.08.97 ##
##########################################################


APPLICATION 	= coverConfigTool
PROJECT	        = General
CATEGORY        = 

# put all object files that are needed to build the module here:

MAIN            = main

OBJECTS         = $(OBJDIR)/$(MAIN).o \
		  $(OBJDIR)/projectionarea.o \
		  $(OBJDIR)/computeprojvalues.o \
		  $(OBJDIR)/host.o \
		  $(OBJDIR)/pipe.o \
		  $(OBJDIR)/window.o \
		  $(OBJDIR)/channel.o \
		  $(OBJDIR)/configfileio.o \
		  $(OBJDIR)/covergeneral.o \
		  $(OBJDIR)/tracking.o \
		  $(OBJDIR)/xmlfileio.o \
		  $(OBJDIR)/coverconfigtool.o \
		  $(OBJDIR)/qmake_image_collection.o \
	          $(OBJDIR)/moc_coverconfigtool.o\
		  $(IDENTOBJ)	

UICDECLS = coverconfigtool.h
UICIMPLS = coverconfigtool.cpp

#QTDIR = /usr/lib/qt-2.3.2
include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

MOC	=  $(COVISE_QTDIR)/bin/moc
UIC     = $(COVISE_QTDIR)/bin/uic
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

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJECTS) 
	  $(CC) $(CPPFLAGS) -o $@ $(OBJECTS) \
                $(QTLIB) -lm


############### general object rule ######################
# can be modified (e.g. for dependencies)
##########################################################

$(OBJDIR)/$(MAIN).o: main.cpp \
	$(OBJDIR)/coverconfigtool.h \
	host.h \
	covergeneral.h \
	pipe.h \
	window.h \
	channel.h \
	projectionarea.h
	$(CC) $(CPPFLAGS)  -I./ -I$(OBJDIR)/ -I$(COVISE_QTDIR)/include -c  main.cpp  -o $(OBJDIR)/$(MAIN).o   
	$(POSTCC)

############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################

$(OBJDIR)/projectionarea.o: projectionarea.cpp projectionarea.h
	$(CC) $(CPPFLAGS) -I$(COVISE_QTDIR)/include -c projectionarea.cpp -o $(OBJDIR)/projectionarea.o
	$(POSTCC)

$(OBJDIR)/computeprojvalues.o: computeprojvalues.cpp computeprojvalues.h
	$(CC) $(CPPFLAGS) -I$(COVISE_QTDIR)/include -c computeprojvalues.cpp -o $(OBJDIR)/computeprojvalues.o
	$(POSTCC)

$(OBJDIR)/host.o: host.cpp host.h \
	pipe.h \
	window.h \
	channel.h \
	projectionarea.h
	$(CC) $(CPPFLAGS) -I$(COVISE_QTDIR)/include -c host.cpp -o $(OBJDIR)/host.o 
	$(POSTCC)

$(OBJDIR)/pipe.o: pipe.cpp pipe.h \
	window.h \
	channel.h \
	projectionarea.h
	$(CC) $(CPPFLAGS) -I$(COVISE_QTDIR)/include -c pipe.cpp -o $(OBJDIR)/pipe.o 
	$(POSTCC)

$(OBJDIR)/window.o: window.cpp window.h \
	channel.h \
	projectionarea.h
	$(CC) $(CPPFLAGS) -I$(COVISE_QTDIR)/include -c window.cpp -o $(OBJDIR)/window.o 
	$(POSTCC)


$(OBJDIR)/channel.o: channel.cpp channel.h \
	projectionarea.h
	$(CC) $(CPPFLAGS) -I$(COVISE_QTDIR)/include -c channel.cpp -o $(OBJDIR)/channel.o 
	$(POSTCC)

$(OBJDIR)/configfileio.o: configfileio.cpp configfileio.h \
	covergeneral.h \
	host.h \
	projectionarea.h \
	pipe.h \
	window.h \
	channel.h 
	$(CC) $(CPPFLAGS) -I$(COVISE_QTDIR)/include -c configfileio.cpp -o $(OBJDIR)/configfileio.o 
	$(POSTCC)

$(OBJDIR)/covergeneral.o: covergeneral.cpp covergeneral.h  \
	host.h \
	projectionarea.h \
	pipe.h \
	window.h \
	channel.h
	$(CC) $(CPPFLAGS) -I$(COVISE_QTDIR)/include -c covergeneral.cpp -o $(OBJDIR)/covergeneral.o 
	$(POSTCC)

$(OBJDIR)/tracking.o: tracking.cpp tracking.h
	$(CC) $(CPPFLAGS) -I$(COVISE_QTDIR)/include -c tracking.cpp -o $(OBJDIR)/tracking.o 
	$(POSTCC)

$(OBJDIR)/xmlfileio.o: xmlfileio.cpp xmlfileio.h \
	covergeneral.h \
	host.h \
	projectionarea.h \
	pipe.h \
	window.h \
	channel.h
	$(CC) $(CPPFLAGS) -I$(COVISE_QTDIR)/include -c xmlfileio.cpp -o $(OBJDIR)/xmlfileio.o 
	$(POSTCC)

$(OBJDIR)/coverconfigtool.h: coverconfigtool.ui host.h \
	covergeneral.h \
	pipe.h \
	window.h \
	channel.h \
	projectionarea.h
	$(UIC) coverconfigtool.ui -o $(OBJDIR)/coverconfigtool.h

$(OBJDIR)/coverconfigtool.cpp: $(OBJDIR)/coverconfigtool.h coverconfigtool.ui host.h \
	covergeneral.h \
	pipe.h \
	window.h \
	channel.h \
	projectionarea.h
	$(UIC) coverconfigtool.ui -i $(OBJDIR)/coverconfigtool.h -o $(OBJDIR)/coverconfigtool.cpp

$(OBJDIR)/coverconfigtool.o: $(OBJDIR)/coverconfigtool.cpp coverconfigtool.ui.h \
	$(OBJDIR)/coverconfigtool.h
	$(CC) $(CPPFLAGS) -I../ -I./ -I$(COVISE_QTDIR)/include -c $(OBJDIR)/coverconfigtool.cpp -o $(OBJDIR)/coverconfigtool.o 
	$(POSTCC)

$(OBJDIR)/moc_coverconfigtool.o: $(OBJDIR)/moc_coverconfigtool.cpp $(OBJDIR)/coverconfigtool.h host.h \
	covergeneral.h \
	pipe.h \
	window.h \
	channel.h \
	projectionarea.h
	$(CC) $(CPPFLAGS) -I./ -I$(COVISE_QTDIR)/include -c $(OBJDIR)/moc_coverconfigtool.cpp -o $(OBJDIR)/moc_coverconfigtool.o 
	$(POSTCC)

$(OBJDIR)/moc_coverconfigtool.cpp: $(MOC) $(OBJDIR)/coverconfigtool.h
	$(MOC) $(OBJDIR)/coverconfigtool.h -o $(OBJDIR)/moc_coverconfigtool.cpp

$(OBJDIR)/qmake_image_collection.o: $(OBJDIR)/qmake_image_collection.cpp
	$(CC) $(CPPFLAGS) -I$(COVISE_QTDIR)/include -c $(OBJDIR)/qmake_image_collection.cpp -o $(OBJDIR)/qmake_image_collection.o 

$(OBJDIR)/qmake_image_collection.cpp: images/back1.png \
	images/back2.png \
	images/back3.png \
	images/back4.png \
	images/bottom1.png \
	images/bottom2.png \
	images/bottom3.png \
	images/bottom4.png \
	images/front1.png \
	images/front2.png \
	images/front3.png \
	images/front4.png \
	images/left1.png \
	images/left2.png \
	images/left3.png \
	images/left4.png \
	images/right1.png \
	images/right2.png \
	images/right3.png \
	images/right4.png \
	images/top1.png \
	images/top2.png \
	images/top3.png \
	images/top4.png \
	images/Logo.png \
	images/fob_image_cropped_400_400.png \
	images/motionstar_400_400.png \
	images/polhemus_image_271_271.png \
	images/polhemus2_image_271_271.png \
	images/no_image.png \
	images/fileopen.png
	$(UIC)  -embed coverConfigTool images/back1.png images/back2.png images/back3.png images/back4.png images/bottom1.png images/bottom2.png images/bottom3.png images/bottom4.png images/front1.png images/front2.png images/front3.png images/front4.png images/left1.png images/left2.png images/left3.png images/left4.png images/right1.png images/right2.png images/right3.png images/right4.png images/top1.png images/top2.png images/top3.png images/top4.png images/fileopen.png images/Logo.png images/fob_image_cropped_400_400.png images/motionstar_400_400.png images/polhemus_image_271_271.png images/polhemus2_image_271_271.png images/no_image.png -o $(OBJDIR)/qmake_image_collection.cpp




uiclean:
	-/bin/rm -rf $(UICIMPLS) $(UICDECLS)
