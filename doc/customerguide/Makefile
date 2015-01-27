
# Makefile for Latex and latex2html
# 
# creates Customer Module Reference Guide only
#DEBUGOUT = outfile
#DEBUGOUT = /dev/tty
DEBUGOUT = /dev/null

LTX = latex

L2H3 =	latex2html							    \
	-transparent -local_icons -no_info -long_title 20 -html_version 4.0 \
	-address "&copy 2005 VISENSO GmbH"        \
	-split 5 -link 0 -nonavigation

L2H2 =	latex2html							    \
	-transparent -local_icons -no_info -long_title 20 -html_version 4.0 \
	-address "&copy 2005 VISENSO GmbH"        \
	-split 6 -link 2 -auto_navigation




all:  clean_customerguide customerguide

debug:
	make DEBUGOUT=/dev/tty

MODULE_BASE = modules


customerguide:  ../pdf/customerguide.pdf customerhtml

#customerguide.dvi: customerguide.tex $(ALL_RG_FILES)

customerguide.dvi: 
	-$(LTX)   customerguide.tex > $(DEBUGOUT) 2>&1
	#makeindex customerguide.idx
	-$(LTX)   customerguide.tex > $(DEBUGOUT) 2>&1

../pdf/customerguide.pdf: customerguide.dvi
	-dvipdf   customerguide.dvi > $(DEBUGOUT) 2>&1
	mv customerguide.pdf ../pdf

ps:  customerguide.dvi
	-dvips    customerguide.dvi > $(DEBUGOUT) 2>&1
	-gzip -f  customerguide.ps

../pdf:
	-mkdir ../pdf


include module.make

clean: clean_customerguide clean_customerhtml

clean_customerguide:
	/bin/rm -f customerguide.dvi
	/bin/rm -f customerguide.pdf
	/bin/rm -f customerguide.toc
	/bin/rm -f customerguide.aux

clean_customerhtml:
	/bin/rm -rf ../online/modules/Customer/*

depend:
	./createTexDep $(MODULE_BASE) > module.make
