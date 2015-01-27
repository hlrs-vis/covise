

DEBUGOUT = /dev/null

LTX = latex

L2H3 =	latex2html	-transparent -local_icons -no_info \
		-long_title 20 -html_version 4.0  -split 5 -link 0 -nonavigation

USERSGUIDE     = hlrs_usersguide
MAPEDITOR      = mapeditor

TEX_FILES =	usersguide.tex		\
		cover/cover.tex			\
		intro/intro.tex			\
		$(MAPEDITOR)/$(MAPEDITOR).tex		\
		renderer/renderer.tex		\
		collab/collab.tex		\
		volumerendering/volumerendering.tex	
      	
ALL_UG_FILES =	intro/*				\
		$(MAPEDITOR)/* /		\
		intro/*               \
		cover/*				\
		volumerendering/*               \
		renderer/*  



all:  clean_usersguide usersguide


debug:
	make DEBUGOUT=/dev/tty


depend: prepare
	echo "nothing to do"
	
   
prepare: usersguide.tex


usersguide.tex: $(USERSGUIDE).tex
	cp $(USERSGUIDE).tex usersguide.tex        


usersguide:  ../pdf/usersguide.pdf 

../pdf/usersguide.pdf: ../pdf usersguide.tex $(TEX_FILES) $(ALL_UG_FILES) 
	pdflatex usersguide
	pdflatex usersguide > $(DEBUGOUT) 2>&1
	pdflatex usersguide > $(DEBUGOUT) 2>&1
	mv usersguide.pdf ../pdf


../pdf:
	-mkdir ../pdf
   

clean: clean_usersguide 

clean_usersguide:
	/bin/rm -f usersguide.*
   
