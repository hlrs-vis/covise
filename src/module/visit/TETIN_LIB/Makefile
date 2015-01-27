
# Makefile for Tetin library

# All common UNIX compiler flags and options are now in
# this central makefile.
include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

LIBSUFFIX = .so
TETIN_LIB_FILE = libTetin$(LIBSUFFIX)

PLATFORM = $(COVISEDIR)/$(ARCHSUFFIX)/bin
PROJECT = Visit

OBJECTS = $(OBJDIR)/coTetin.o \
			 $(OBJDIR)/coTetinCommand.o \
			 $(OBJDIR)/coTetin__OutputInterf.o \
			 $(OBJDIR)/coTetin__configDir.o \
          $(OBJDIR)/coTetin__thinCut.o \
          $(OBJDIR)/coTetin__affix.o \
          $(OBJDIR)/coTetin__defCurve.o \
          $(OBJDIR)/coTetin__defDensPoly.o \
          $(OBJDIR)/coTetin__defModel.o \
          $(OBJDIR)/coTetin__defSurf.o \
          $(OBJDIR)/coTetin__matPoint.o \
          $(OBJDIR)/coTetin__mergeSurf.o \
          $(OBJDIR)/coTetin__period.o \
          $(OBJDIR)/coTetin__transl.o \
          $(OBJDIR)/coTetin__prescPnt.o \
          $(OBJDIR)/coTetin__getprescPnt.o \
          $(OBJDIR)/coTetin__rdFamBoco.o \
          $(OBJDIR)/coTetin__bocoFile.o \
          $(OBJDIR)/coTetin__trianTol.o \
          $(OBJDIR)/coTetin__tetinFile.o \
          $(OBJDIR)/coTetin__replayFile.o \
          $(OBJDIR)/coTetin__Hexa.o \
          $(OBJDIR)/coTetin__trianFam.o \
          $(OBJDIR)/coTetin__apprxCurve.o \
          $(OBJDIR)/coTetin__transGeom.o \
          $(OBJDIR)/coTetin__delCurve.o \
          $(OBJDIR)/coTetin__delSurf.o \
          $(OBJDIR)/coTetin__delFam.o \
          $(OBJDIR)/coTetin__Proj.o \
          $(OBJDIR)/coTetin__utils.o


all:   $(OBJDIR) $(PLATFORM)/$(TETIN_LIB_FILE)

$(PLATFORM)/$(TETIN_LIB_FILE): $(OBJECTS) 
	$(MKSHARED)$(TETIN_LIB_FILE) $(OBJDIR)/*.o $(ADDOBJECTS)
	chmod 755 $(TETIN_LIB_FILE)
	cp $(TETIN_LIB_FILE) $(PLATFORM)
	@echo tetin_lib created

$(OBJDIR):
	mkdir $(OBJDIR)


clean:
	-/bin/rm -rf $(OBJDIR) $(PLATFORM)/$(TETIN_LIB_FILE) $(EXTRACLEAN)

THISDIR = src/covise_lib
SRCFILES = $(THISDIR)/*.cpp $(THISDIR)/*.h $(THISDIR)/Makefile
DEVFILES = $(THISDIR)/*.h


tar_bin_modules:
	cd $(COVISEDIR) ; \
	if (test -f co$(ARCHSUFFIX).modules.$(PROJECT).tar) \
	then tar uvf co$(ARCHSUFFIX).modules.$(PROJECT).tar $(ARCHSUFFIX)/bin/$(TETIN_LIB_FILE);\
	else tar cvf co$(ARCHSUFFIX).modules.$(PROJECT).tar $(ARCHSUFFIX)/bin/$(TETIN_LIB_FILE);\
	fi

tar_src_core:
	cd $(COVISEDIR) ; \
	if (test -f cosrcc.tar) \
	then tar uvf cosrcc.tar $(SRCFILES);\
	else tar cvf cosrcc.tar $(SRCFILES);\
	fi


tar_dev:
	cd $(COVISEDIR) ; \
	if (test -f codevh.tar) \
	then tar uvf codevh.tar $(DEVFILES);\
	else tar cvf codevh.tar $(DEVFILES);\
	fi
	
depend:
	$(MAKEDEPEND)


$(OBJDIR)/coTetin.o: coTetin.cpp coTetin.h
	$(LIBPRECC) coTetin.cpp
	$(LIBCC)coTetin.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetinCommand.o: coTetinCommand.cpp coTetinCommand.h
	$(LIBPRECC) coTetinCommand.cpp
	$(LIBCC)coTetinCommand.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__OutputInterf.o: coTetin__OutputInterf.cpp coTetin__OutputInterf.h
	$(LIBPRECC) coTetin__OutputInterf.cpp
	$(LIBCC)coTetin__OutputInterf.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__configDir.o: coTetin__configDir.cpp coTetin__configDir.h
	$(LIBPRECC) coTetin__configDir.cpp
	$(LIBCC)coTetin__configDir.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__thinCut.o: coTetin__thinCut.cpp coTetin__thinCut.h
	$(LIBPRECC) coTetin__thinCut.cpp
	$(LIBCC)coTetin__thinCut.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__affix.o: coTetin__affix.cpp coTetin__affix.h
	$(LIBPRECC) coTetin__affix.cpp
	$(LIBCC)coTetin__affix.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__defCurve.o: coTetin__defCurve.cpp coTetin__defCurve.h
	$(LIBPRECC) coTetin__defCurve.cpp
	$(LIBCC)coTetin__defCurve.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__defDensPoly.o: coTetin__defDensPoly.cpp coTetin__defDensPoly.h
	$(LIBPRECC) coTetin__defDensPoly.cpp
	$(LIBCC)coTetin__defDensPoly.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__defModel.o: coTetin__defModel.cpp coTetin__defModel.h
	$(LIBPRECC) coTetin__defModel.cpp
	$(LIBCC)coTetin__defModel.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__defSurf.o: coTetin__defSurf.cpp coTetin__defSurf.h
	$(LIBPRECC) coTetin__defSurf.cpp
	$(LIBCC)coTetin__defSurf.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__mergeSurf.o: coTetin__mergeSurf.cpp coTetin__mergeSurf.h
	$(LIBPRECC) coTetin__mergeSurf.cpp
	$(LIBCC)coTetin__mergeSurf.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__period.o: coTetin__period.cpp coTetin__period.h
	$(LIBPRECC) coTetin__period.cpp
	$(LIBCC)coTetin__period.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__transl.o: coTetin__transl.cpp coTetin__transl.h
	$(LIBPRECC) coTetin__transl.cpp
	$(LIBCC)coTetin__transl.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__rdFamBoco.o: coTetin__rdFamBoco.cpp coTetin__rdFamBoco.h
	$(LIBPRECC) coTetin__rdFamBoco.cpp
	$(LIBCC)coTetin__rdFamBoco.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__bocoFile.o: coTetin__bocoFile.cpp coTetin__bocoFile.h
	$(LIBPRECC) coTetin__bocoFile.cpp
	$(LIBCC)coTetin__bocoFile.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__trianTol.o: coTetin__trianTol.cpp coTetin__trianTol.h
	$(LIBPRECC) coTetin__trianTol.cpp
	$(LIBCC)coTetin__trianTol.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__tetinFile.o: coTetin__tetinFile.cpp coTetin__tetinFile.h
	$(LIBPRECC) coTetin__tetinFile.cpp
	$(LIBCC)coTetin__tetinFile.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__replayFile.o: coTetin__replayFile.cpp coTetin__replayFile.h
	$(LIBPRECC) coTetin__replayFile.cpp
	$(LIBCC)coTetin__replayFile.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__Hexa.o: coTetin__Hexa.cpp coTetin__Hexa.h
	$(LIBPRECC) coTetin__Hexa.cpp
	$(LIBCC)coTetin__Hexa.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__trianFam.o: coTetin__trianFam.cpp coTetin__trianFam.h
	$(LIBPRECC) coTetin__trianFam.cpp
	$(LIBCC)coTetin__trianFam.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__apprxCurve.o: coTetin__apprxCurve.cpp coTetin__apprxCurve.h
	$(LIBPRECC) coTetin__apprxCurve.cpp
	$(LIBCC)coTetin__apprxCurve.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__transGeom.o: coTetin__transGeom.cpp coTetin__transGeom.h
	$(LIBPRECC) coTetin__transGeom.cpp
	$(LIBCC)coTetin__transGeom.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__delCurve.o: coTetin__delCurve.cpp coTetin__delCurve.h
	$(LIBPRECC) coTetin__delCurve.cpp
	$(LIBCC)coTetin__delCurve.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__delSurf.o: coTetin__delSurf.cpp coTetin__delSurf.h
	$(LIBPRECC) coTetin__delSurf.cpp
	$(LIBCC)coTetin__delSurf.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__delFam.o: coTetin__delFam.cpp coTetin__delFam.h
	$(LIBPRECC) coTetin__delFam.cpp
	$(LIBCC)coTetin__delFam.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__Proj.o: coTetin__Proj.cpp coTetin__Proj.h
	$(LIBPRECC) coTetin__Proj.cpp
	$(LIBCC)coTetin__Proj.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__utils.o: coTetin__utils.cpp coTetin__utils.h
	$(LIBPRECC) coTetin__utils.cpp
	$(LIBCC)coTetin__utils.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__prescPnt.o: coTetin__prescPnt.cpp coTetin__prescPnt.h
	$(LIBPRECC) coTetin__prescPnt.cpp
	$(LIBCC)coTetin__prescPnt.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__getprescPnt.o: coTetin__prescPnt.cpp coTetin__prescPnt.h
	$(LIBPRECC) coTetin__getprescPnt.cpp
	$(LIBCC)coTetin__getprescPnt.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

$(OBJDIR)/coTetin__matPoint.o: coTetin__matPoint.cpp coTetin__matPoint.h
	$(LIBPRECC) coTetin__matPoint.cpp
	$(LIBCC)coTetin__matPoint.cpp $(LIBCPPFLAGS) 
	$(POSTCC)

# DO NOT DELETE

$(OBJDIR)/coTetin.o: coTetin.h coTetin__thinCut.h coTetinCommand.h
$(OBJDIR)/coTetin.o: coTetin__affix.h coTetin__defCurve.h coTetin__utils.h
$(OBJDIR)/coTetin.o: coTetin__apprxCurve.h coTetin__getprescPnt.h
$(OBJDIR)/coTetin.o: coTetin__transGeom.h coTetin__defDensPoly.h
$(OBJDIR)/coTetin.o: coTetin__defModel.h coTetin__defSurf.h
$(OBJDIR)/coTetin.o: coTetin__matPoint.h coTetin__mergeSurf.h
$(OBJDIR)/coTetin.o: coTetin__period.h coTetin__transl.h coTetin__rdFamBoco.h
$(OBJDIR)/coTetin.o: coTetin__bocoFile.h coTetin__prescPnt.h
$(OBJDIR)/coTetin.o: coTetin__trianTol.h coTetin__tetinFile.h
$(OBJDIR)/coTetin.o: coTetin__replayFile.h coTetin__configDir.h
$(OBJDIR)/coTetin.o: coTetin__Hexa.h coTetin__OutputInterf.h
$(OBJDIR)/coTetin.o: coTetin__trianFam.h coTetin__delFam.h coTetin__delSurf.h
$(OBJDIR)/coTetin.o: coTetin__delCurve.h coTetin__Proj.h
$(OBJDIR)/coTetinCommand.o: coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__Hexa.o: coTetin__Hexa.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__OutputInterf.o: coTetin__OutputInterf.h coTetinCommand.h
$(OBJDIR)/coTetin__OutputInterf.o: coTetin.h
$(OBJDIR)/coTetin__Proj.o: coTetin__Proj.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__affix.o: coTetin__affix.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__apprxCurve.o: coTetin__apprxCurve.h coTetinCommand.h
$(OBJDIR)/coTetin__apprxCurve.o: coTetin.h
$(OBJDIR)/coTetin__bocoFile.o: coTetin__bocoFile.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__configDir.o: coTetin__configDir.h coTetinCommand.h
$(OBJDIR)/coTetin__configDir.o: coTetin.h
$(OBJDIR)/coTetin__defCurve.o: coTetin__defCurve.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__defCurve.o: coTetin__utils.h
$(OBJDIR)/coTetin__defDensPoly.o: coTetin__defDensPoly.h coTetinCommand.h
$(OBJDIR)/coTetin__defDensPoly.o: coTetin.h
$(OBJDIR)/coTetin__defModel.o: coTetin__defModel.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__defSurf.o: coTetin__defSurf.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__defSurf.o: coTetin__utils.h
$(OBJDIR)/coTetin__delCurve.o: coTetin__delCurve.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__delFam.o: coTetin__delFam.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__delSurf.o: coTetin__delSurf.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__getprescPnt.o: coTetin__getprescPnt.h coTetinCommand.h
$(OBJDIR)/coTetin__getprescPnt.o: coTetin.h
$(OBJDIR)/coTetin__matPoint.o: coTetin__matPoint.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__mergeSurf.o: coTetin__mergeSurf.h coTetinCommand.h
$(OBJDIR)/coTetin__mergeSurf.o: coTetin.h
$(OBJDIR)/coTetin__period.o: coTetin__period.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__prescPnt.o: coTetin__prescPnt.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__rdFamBoco.o: coTetin__rdFamBoco.h coTetinCommand.h
$(OBJDIR)/coTetin__rdFamBoco.o: coTetin.h
$(OBJDIR)/coTetin__replayFile.o: coTetin__replayFile.h coTetinCommand.h
$(OBJDIR)/coTetin__replayFile.o: coTetin.h
$(OBJDIR)/coTetin__tetinFile.o: coTetin__tetinFile.h coTetinCommand.h
$(OBJDIR)/coTetin__tetinFile.o: coTetin.h
$(OBJDIR)/coTetin__thinCut.o: coTetin__thinCut.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__transGeom.o: coTetin__transGeom.h coTetinCommand.h
$(OBJDIR)/coTetin__transGeom.o: coTetin.h
$(OBJDIR)/coTetin__transl.o: coTetin__transl.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__trianFam.o: coTetin__trianFam.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__trianTol.o: coTetin__trianTol.h coTetinCommand.h coTetin.h
$(OBJDIR)/coTetin__utils.o: coTetin__utils.h coTetinCommand.h coTetin.h
