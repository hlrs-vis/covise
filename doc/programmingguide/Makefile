

DEBUGOUT = /dev/null

LTX = latex

L2H3 =	latex2html	-transparent -local_icons -no_info \
		-long_title 20 -html_version 4.0  -split 5 -link 0 -nonavigation

PROGRAMMINGGUIDE     = hlrs_programmingguide

DOXYTEX_FILES = coVRPluginSupport/classco_v_r_plugin_support.tex \
				coVRPluginSupport/classco_pointer_button.tex

TEX_FILES = programmingguide.tex  \
            Basics/Basics.tex \
            ModuleProgramming/ModuleProgramming.tex \
            CoviseDataObjects/CoviseDataObjects.tex \
            SimulationLibrary/SimulationLibrary.tex \
            OtherFeatures/OtherFeatures.tex \
            COVERPlugin/COVERPlugin.tex \
            ExampleModules/ExampleModules.tex \
            QuickReference/QuickReference.tex \
			${DOXYTEX_FILES}


all: programmingguide 


debug:
	make DEBUGOUT=/dev/tty


depend: prepare
	echo "Nothing to do."
	
   
prepare: programmingguide.tex ../pdf

${DOXYTEX_FILES}: coVRPluginSupport.doxyfile ../../src/OpenCOVER/cover/coVRPluginSupport.h
	doxygen coVRPluginSupport.doxyfile

programmingguide.tex: $(PROGRAMMINGGUIDE).tex
	cp $(PROGRAMMINGGUIDE).tex programmingguide.tex        


programmingguide:  ../pdf/programmingguide.pdf 


../pdf/programmingguide.pdf: prepare $(TEX_FILES)
	pdflatex programmingguide
	makeindex programmingguide.idx > $(DEBUGOUT)
	pdflatex programmingguide > $(DEBUGOUT) 2>&1
	pdflatex programmingguide > $(DEBUGOUT) 2>&1
	mv programmingguide.pdf ../pdf


../pdf:
	-mkdir ../pdf
   
clean: clean_programmingguide 


clean_programmingguide:
	${RM} programmingguide.*
	${RM} ${DOXYTEX_FILES}
