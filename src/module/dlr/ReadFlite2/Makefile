#COMPILE_FOR=$(shell basename $(QMAKESPEC) | cut -f1 -d-)

#DIRECTORY=$(shell basename `pwd`)
QMAKE=$(QT_HOME)/bin/qmake

MAKEFILE =      Makefile.$(FRAMEWORK)-$(ARCHSUFFIX)


first: all

#$(MAKECMDGOALS): $(MAKEFILE)
#	$(MAKE) -f $(MAKEFILE) QMAKE=$(QMAKE) $(MAKECMDGOALS)

makefile: $(MAKEFILE)

all: makefile 
	$(MAKE) -f $(MAKEFILE) QMAKE=$(QMAKE)

clean: makefile
	$(MAKE) -f $(MAKEFILE) clean QMAKE=$(QMAKE)
	$(RM) -r objects_$(FRAMEWORK)-$(ARCHSUFFIX)
	$(RM) -r moc_$(FRAMEWORK)-$(ARCHSUFFIX)
	$(RM) $(MAKEFILE)

tar_bin_core tar_bin_modules tar_bin_plugins tar_dev \
tar_dev_examples tar_src_core tar_src_modules tar \
src_plugins: makefile
	$(MAKE) -f $(MAKEFILE) QMAKE=$(QMAKE) $@

makefiles: depend
	$(MAKE) -f $(MAKEFILE) QMAKE=$(QMAKE) $@

depend:
	$(RM) $(MAKEFILE)
	$(QMAKE) -o $(MAKEFILE)


$(MAKEFILE): $(QMAKESPEC)/*.conf $(QMAKESPEC)/../common.conf $(QMAKESPEC)/../*.pro *.pro
	$(QMAKE) -o $(MAKEFILE)

.NO_WINK_IN: $(MAKEFILE)
.NO_CMP_NON_MF_DEPS: $(MAKEFILE)
.NO_CMP_SCRIPT: $(MAKEFILE)
