
# Makefile for Tools

# All common UNIX compiler flags and options are now in
# this central makefile.
include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)
include $(COVISEDIR)/src/make.rules
all: $(PLATFORM)/bin/covdvi $(PLATFORM)/bin/coviseconfig2tcl
$(PLATFORM)/bin/covdvi: covdvi.sh
	cp -rf covdvi.sh $(PLATFORM)/bin/covdvi
	chmod 755  $(PLATFORM)/bin/covdvi

$(PLATFORM)/bin/coviseconfig2tcl:
	(cd configureCovise ; $(MAKE))
