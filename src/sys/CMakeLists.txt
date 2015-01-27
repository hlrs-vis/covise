# @file
# 
# CMakeLists.txt for covise sys
#
# @author Blasius Czink

PROJECT(sys CXX C)

ADD_SUBDIRECTORY(controller)
#ADD_SUBDIRECTORY(convertconfig)
ADD_SUBDIRECTORY(carSound)
ADD_SUBDIRECTORY(crb)
ADD_SUBDIRECTORY(vrb)
ADD_SUBDIRECTORY(RemoteDaemon)
ADD_SUBDIRECTORY(AccessGridDaemon)
ADD_SUBDIRECTORY(CoviseSSLDaemon)
ADD_SUBDIRECTORY(mapeditor)
ADD_SUBDIRECTORY(tabletUI)
ADD_SUBDIRECTORY(GuiRenderMessage)
ADD_SUBDIRECTORY(ScriptingInterface)
ADD_SUBDIRECTORY(wsinterface)

#  ADD_SUBDIRECTORY(daemon)
