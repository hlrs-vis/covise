
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


# PYTHONPATH is python-module-search-path.
PYTHONPATH := $(COVISEDIR)/Python:$(COVISEDIR)/$(ARCHSUFFIX)/bin:$(PYTHONPATH):$(COVISEDIR)/src/application/ui/auxils:$(COVISEDIR)/src/application/ui/vr-prepare/negotiator:$(COVISEDIR)/src/application/ui/vr-prepare

all: unittests

unittests:
	$(COVISE_PYTHON) test_all.py


cleanforgrep:
	-rm *.pyc
	-rm *~

clean: cleanforgrep

# eof
