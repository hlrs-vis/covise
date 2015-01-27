
DEBUGOUT = /dev/null

LTX = latex

L2H3 =	latex2html -transparent -local_icons -no_info -long_title 20 -html_version 4.0  -split 5 -link 0 -nonavigation

REFGUIDE     = hlrs_refguide

MODULE_BASE = modules


-include refguide.make


all:  clean_refguide refguide


debug:
	make DEBUGOUT=/dev/tty

refguide:  ../pdf/refguide.pdf 


refguide.tex: $(REFGUIDE).tex
	cp $(REFGUIDE).tex refguide.tex   


../pdf/refguide.pdf: ../pdf tex refguide.tex
	pdflatex refguide
	pdflatex refguide > $(DEBUGOUT) 2>&1
	pdflatex refguide > $(DEBUGOUT) 2>&1
	mv refguide.pdf ../pdf


../pdf:
	-mkdir ../pdf



clean: clean_refguide 


clean_refguide:
	/bin/rm -f refguide.*



tex: modulelist.txt
	#${RM} modules.tex modulelist.txt
	#cd ../../src/module && $(MAKE) pdf
	#cd ../../src/renderer && $(MAKE) pdf
	../scripts/makemodulestex.sh > modules.tex
  
