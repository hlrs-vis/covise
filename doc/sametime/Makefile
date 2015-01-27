# Makefile for Latex and latex2html
# creates NS3 (COVISE Conference Room - Sametime)
# create separate online directory for NS3 in online_doc

# add file name here if you want to get all the output from the HTML 
#DEBUGOUT = outfile
#DEBUGOUT = /dev/tty
DEBUGOUT = /dev/null

LTX = latex

L2H3 = latex2html -transparent -split 5 -link 0 -local_icons \
       -address "&copy 2001 VirCinity IT GmbH Stuttgart" \
       -nonavigation -long_title 20 -no_info \
       -html_version 4.0

L2H2 = latex2html -transparent -split 6 -link 2 -local_icons \
       -address "&copy 2001 VirCinity IT GmbH Stuttgart" \
       -auto_navigation -no_info -long_title 20 \
      -html_version 4.0


TEX_FILES1 = sametime.tex  \
Summary/Summary.tex \
GettingStarted/GettingStarted.tex \
ConnectList/ConnectList.tex \
Whiteboard/Whiteboard.tex \
SharedApplication/SharedApplication.tex \
CollabVisual/CollabVisual.tex \
ChatMessages/ChatMessages.tex \
AudioVideo/AudioVideo.tex \
UsageHints/UsageHints.tex

all: sametimehtml ../pdf/sametime.pdf
debug:
	make DEBUGOUT=/dev/tty

depend:
	echo "nothing to do"

sametime: sametime.dvi ../pdf/sametime.pdf sametimehtml

sametime.dvi: sametime.tex $(TEX_FILES)
	$(LTX) sametime.tex > $(DEBUGOUT) 2>&1
	$(LTX) sametime.tex > $(DEBUGOUT) 2>&1

sametime.ps:  sametime.dvi
	dvips   sametime.dvi
	gzip -f sametime.ps
   
../pdf/sametime.pdf: sametime.dvi ../pdf
	dvipdf   sametime.dvi
	mv sametime.pdf ../pdf

../pdf:
	-mkdir ../pdf
		      
sametimehtml:      ../online_doc/sametime/chapter/Summary/Summary.html
sametime.dvi: Summary/Summary.tex

sametimehtml:      ../online_doc/sametime/chapter/GettingStarted/Getting_Started.html
sametime.dvi: GettingStarted/GettingStarted.tex

sametimehtml:      ../online_doc/sametime/chapter/ConnectList/Connect_List.html
sametime.dvi: ConnectList/ConnectList.tex

sametimehtml:      ../online_doc/sametime/chapter/Whiteboard/Whiteboard.html
sametime.dvi: Whiteboard/Whiteboard.tex

sametimehtml:      ../online_doc/sametime/chapter/SharedApplication/Shared_Application.html
sametime.dvi: SharedApplication/SharedApplication.tex

sametimehtml:      ../online_doc/sametime/chapter/ChatMessages/Chat_Messages.html
sametime.dvi: ChatMessages/ChatMessages.tex

sametimehtml:      ../online_doc/sametime/chapter/AudioVideo/Audio_and_Video.html
sametime.dvi: AudioVideo/AudioVideo.tex

sametimehtml:      ../online_doc/sametime/chapter/CollabVisual/CollabVisual.html
sametime.dvi: CollabVisual/CollabVisual.tex

sametimehtml:      ../online_doc/sametime/chapter/UsageHints/Usage_Hints.html
sametime.dvi: UsageHints/UsageHints.tex


../online_doc/sametime/chapter/Summary/Summary.html: Summary/Summary.tex	
	-cp Summary/*.gif  ../online_doc/sametime/chapter/Summary
	$(L2H3) -address "&copy 2001 VirCinity IT GmbH Stuttgart, Author Sven Kufer / Peter Wolf" -dir ../online_doc/sametime/chapter/Summary Summary/Summary.tex > $(DEBUGOUT) 2>&1

../online_doc/sametime/chapter/GettingStarted/Getting_Started.html: GettingStarted/GettingStarted.tex	
	-cp GettingStarted/*.gif  ../online_doc/sametime/chapter/GettingStarted
	$(L2H3) -address "&copy 2001 VirCinity IT GmbH Stuttgart, Author Sven Kufer / Peter Wolf" -dir ../online_doc/sametime/chapter/GettingStarted
	$(L2H3) GettingStarted/GettingStarted.tex > $(DEBUGOUT) 2>&1

../online_doc/sametime/chapter/ConnectList/Connect_List.html: ConnectList/ConnectList.tex	
	-cp ConnectList/*.gif  ../online_doc/sametime/chapter/ConnectList
	$(L2H3) -address "&copy 2001 VirCinity IT GmbH Stuttgart, Author Sven Kufer / Peter Wolf" -dir ../online_doc/sametime/chapter/ConnectList ConnectList/ConnectList.tex > $(DEBUGOUT) 2>&1

../online_doc/sametime/chapter/Whiteboard/Whiteboard.html: Whiteboard/Whiteboard.tex	
	-cp Whiteboard/*.gif  ../online_doc/sametime/chapter/Whiteboard
	$(L2H3) -address "&copy 2001 VirCinity IT GmbH Stuttgart, Author Sven Kufer / Peter Wolf" -dir ../online_doc/sametime/chapter/Whiteboard Whiteboard/Whiteboard.tex > $(DEBUGOUT) 2>&1

../online_doc/sametime/chapter/SharedApplication/Shared_Application.html: SharedApplication/SharedApplication.tex	
	-cp SharedApplication/*.gif  ../online_doc/sametime/chapter/SharedApplication
	$(L2H3) -address "&copy 2001 VirCinity IT GmbH Stuttgart, Author Sven Kufer / Peter Wolf" -dir ../online_doc/sametime/chapter/SharedApplication SharedApplication/SharedApplication.tex > $(DEBUGOUT) 2>&1

../online_doc/sametime/chapter/ChatMessages/Chat_Messages.html: ChatMessages/ChatMessages.tex	
	-cp ChatMessages/*.gif  ../online_doc/sametime/chapter/ChatMessages
	$(L2H3) -address "&copy 2001 VirCinity IT GmbH Stuttgart, Author Sven Kufer / Peter Wolf" -dir ../online_doc/sametime/chapter/ChatMessages ChatMessages/ChatMessages.tex > $(DEBUGOUT) 2>&1

../online_doc/sametime/chapter/AudioVideo/Audio_and_Video.html: AudioVideo/AudioVideo.tex	
	-cp AudioVideo/*.gif  ../online_doc/sametime/chapter/AudioVideo
	$(L2H3) -address "&copy 2001 VirCinity IT GmbH Stuttgart, Author Sven Kufer / Peter Wolf" -dir ../online_doc/sametime/chapter/AudioVideo AudioVideo/AudioVideo.tex > $(DEBUGOUT) 2>&1
	
../online_doc/sametime/chapter/CollabVisual/CollabVisual.html: CollabVisual/CollabVisual.tex	
	-cp CollabVisual/*.gif  ../online_doc/sametime/chapter/CollabVisual
	$(L2H3) -address "&copy 2001 VirCinity IT GmbH Stuttgart, Author Sven Kufer / Peter Wolf" -dir ../online_doc/sametime/chapter/CollabVisual CollabVisual/CollabVisual.tex > $(DEBUGOUT) 2>&1

../online_doc/sametime/chapter/UsageHints/Usage_Hints.html: UsageHints/UsageHints.tex	
	-cp UsageHints/*.gif  ../online_doc/sametime/chapter/UsageHints
	$(L2H3) -address "&copy 2001 VirCinity IT GmbH Stuttgart, Author Sven Kufer / Peter Wolf" -dir ../online_doc/sametime/chapter/UsageHints UsageHints/UsageHints.tex > $(DEBUGOUT) 2>&1

clean : clean_sametime clean_sametimehtml

clean_sametime:
	/bin/rm -f sametime.dvi 
	/bin/rm -f sametime.ps 
	/bin/rm -f sametime.pdf
	/bin/rm -f sametime.toc
	/bin/rm -f sametime.aux

clean_sametimehtml:  
	/bin/rm -f ../online_doc/sametime/chapter/Summary/*  
	/bin/rm -f ../online_doc/sametime/chapter/GettingStarted/* 	
	/bin/rm -f ../online_doc/sametime/chapter/ConnectList/*
	/bin/rm -f ../online_doc/sametime/chapter/Whiteboard/* 
	/bin/rm -f ../online_doc/sametime/chapter/SharedApplication/*
	/bin/rm -f ../online_doc/sametime/chapter/ChatMessages/*
	/bin/rm -f ../online_doc/sametime/chapter/AudioVideo/*
	/bin/rm -f ../online_doc/sametime/chapter/CollabVisual/*
	/bin/rm -f ../online_doc/sametime/chapter/UsageHints/*
	
