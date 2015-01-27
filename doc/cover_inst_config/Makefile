
DEBUGOUT = /dev/null

LTX = latex

L2H3 =	latex2html	-transparent -local_icons -no_info \
		-long_title 20 -html_version 4.0  -split 5 -link 0 -nonavigation

INST     = hlrs_cover_inst_config


TEX_FILES =	cover_inst_config.tex \
               display/*	\
		         input/* 	\
		         introduction/*	


all:  clean_cover_inst_config cover_inst_config 


debug:
	make DEBUGOUT=/dev/tty


depend:
	echo "do nothing"


prepare: cover_inst_config.tex


cover_inst_config.tex: $(INST).tex
	cp $(INST).tex cover_inst_config.tex  


cover_inst_config:  ../pdf/cover_inst_config.pdf


../pdf/cover_inst_config.pdf: ../pdf cover_inst_config.tex $(TEX_FILES)   
	pdflatex cover_inst_config
	pdflatex cover_inst_config > $(DEBUGOUT) 2>&1
	pdflatex cover_inst_config > $(DEBUGOUT) 2>&1
	mv cover_inst_config.pdf ../pdf


../pdf:
	-mkdir ../pdf


clean: clean_cover_inst_config 

clean_cover_inst_config:
	/bin/rm -f cover_inst_config.*
