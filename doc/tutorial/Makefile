
DEBUGOUT = /dev/null

LTX = latex

L2H3 =	latex2html	-transparent -local_icons -no_info \
		-long_title 20 -html_version 4.0  -split 5 -link 0 -nonavigation

TUTORIAL     = hlrs_tutorial

TEX_FILES    = tutorial.tex  \
             Preface/Preface.tex \
             StartCOVISE/StartCOVISE.tex \
             Using_the_Renderer/Using_the_Renderer.tex \
             Working_with_Modules/Working_with_Modules.tex \
             Analysis_of_3D_Data/Analysis_of_3D_Data.tex \
             AdvancedTopics/AdvancedTopics.tex




all:  clean_tutorial tutorial


debug:
	make DEBUGOUT=/dev/tty


depend: prepare
	echo "nothing to do"
	
   
prepare: tutorial.tex


tutorial.tex: $(TUTORIAL).tex
	cp $(TUTORIAL).tex tutorial.tex        


tutorial:  ../pdf/tutorial.pdf 
   

../pdf/tutorial.pdf: ../pdf tutorial.tex $(TEX_FILES)  
	pdflatex tutorial
	pdflatex tutorial > $(DEBUGOUT) 2>&1
	pdflatex tutorial > $(DEBUGOUT) 2>&1
	mv tutorial.pdf ../pdf


../pdf:
	-mkdir ../pdf


clean: clean_tutorial 


clean_tutorial:
	/bin/rm -f tutorial.* 
