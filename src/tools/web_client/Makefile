
# Makefile for 

# All common UNIX compiler flags and options are now in
# this central makefile.


OBJ =	WWWReadThread.class    \
        WWWConnection.class    \
        UserInterface.class    \
        PlugIn.class           \
        SceneGraph.class       \
        ClientApplet.class


default: all 

all:   $(OBJ)
	cp *.class ../../sgin32/bin/web_client/

ClientApplet.class: ClientApplet.java
	javac ClientApplet.java

WWWConnection.class: WWWConnection.java
	javac WWWConnection.java

UserInterface.class: UserInterface.java
	javac UserInterface.java
        
WWWReadThread.class: WWWReadThread.java
	javac WWWReadThread.java

PlugIn.class: PlugIn.java
	javac PlugIn.java

SceneGraph.class: SceneGraph.java
	javac SceneGraph.java


clean:
	rm *.class

clobber: clean


