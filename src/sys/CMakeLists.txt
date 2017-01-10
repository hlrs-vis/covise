# @file
# 
# CMakeLists.txt for covise sys
#
# @author Blasius Czink

PROJECT(sys CXX C)

if (NOT COVISE_BUILD_ONLY_COVER_PREREQ)
   ADD_SUBDIRECTORY(controller)
   ADD_SUBDIRECTORY(crb)
   ADD_SUBDIRECTORY(mapeditor)
   ADD_SUBDIRECTORY(ScriptingInterface)
   ADD_SUBDIRECTORY(wsinterface)
   ADD_SUBDIRECTORY(AccessGridDaemon)
endif (NOT COVISE_BUILD_ONLY_COVER_PREREQ)
#ADD_SUBDIRECTORY(convertconfig)
ADD_SUBDIRECTORY(carSound)
ADD_SUBDIRECTORY(vrb)
ADD_SUBDIRECTORY(RemoteDaemon)
ADD_SUBDIRECTORY(CoviseSSLDaemon)
ADD_SUBDIRECTORY(tabletUI)
ADD_SUBDIRECTORY(GuiRenderMessage)

#  ADD_SUBDIRECTORY(daemon)
