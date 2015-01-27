
# Copyright (c) 2006-2007 Visenso GmbH

all: unittests

unittests:
	cd unittests/ && $(MAKE) unittests

clean:
	$(RM) *.pyc
	cd import && $(MAKE) clean

.PHONY: clean unittests

# eof
