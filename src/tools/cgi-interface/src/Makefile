# -----------------------------------------------------------------------------
# 23456789012345678901234567890123456789012345678901234567890123456789012345678
# -----------------------------------------------------------------------------

### ---------------------------------------------------------------------------
###	        Das Makefile zum OPA.  (Online Projekt Antrag)
### ---------------------------------------------------------------------------


# wo befindet sich das CGI-Verzeichnis. Dorthin wird dieses Makefile das
# Executable kopieren.

CGIDIR          = /serv/services/cgi-vis/cgi-bin/


LOGFILE		= \"/serv/services/cgi-vis/log/lic.log\"
RH		= \"/serv/services/cgi-vis/lib/reply_header\"
RF		= \"/serv/services/cgi-vis/lib/reply_footer\"

### Compilier-Optionen ###

CCOPTS          = -Wall -g
CC              = gcc
LNOPTS          = -lc

### OBJECTS ###
OBJECTS         = getcgi.o coLicense.o makeLicense.o
OBJECTSTEST     = getcgi.o testit.o makeLicense.o


### Rules zum Compilieren ###
coLicense:	$(OBJECTS)
		@echo "===> Linke das Programm: coLicense"
		@$(CC) -o coLicense $(OBJECTS) $(LNOPTS)

		@echo "---> Copiere den neuen coLicense in sein CGI-BIN."
		@cp coLicense $(CGIDIR)

makeLicense.o:	makeLicense.c
		@echo "===> Compiliere das Programm: makeLicense.c"
		@$(CC) -c makeLicense.c $(CCOPTS)

getcgi.o:	getcgi.c getcgi.h parse.h
		@echo "===> Compiliere das Programm: getcgi.c"
		@$(CC) -c getcgi.c $(CCOPTS)

coLicense.o:	coLicense.c getcgi.h
		@echo "===> Compiliere das Programm: coLicense.c"
		@$(CC) -c coLicense.c $(CCOPTS) -DLOCKFILE=$(LOCKFILE) \
					  -DREPLY_HEADER=$(RH)   \
					  -DREPLY_FOOTER=$(RF)   \
					  -DLOGFILE=$(LOGFILE)
