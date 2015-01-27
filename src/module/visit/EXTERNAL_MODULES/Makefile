# Makefile for modules without source code
CATEGORY = Visit
PROJECT = Visit
MODULES = IcemTessellate IcemHexa IcemProject

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

all: $(PLATFORM)/bin/$(CATEGORY)
	cp -f $(MODULES) $(PLATFORM)/bin/$(CATEGORY)

include $(COVISEDIR)/src/make.rules

BINFILES=$${THISDIR}/*Icem*
SRCFILES=

$(PLATFORM)/bin/$(CATEGORY):
	-mkdir -p $(PLATFORM)/bin/$(CATEGORY)
