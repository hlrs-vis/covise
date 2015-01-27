# @file
# 
# CMakeLists.txt for kernel - vrml - defuse
#
# @author Blasius Czink

SET(DEFUSE_SOURCES
  defuse.cpp
)

ADD_COVISE_EXECUTABLE(defuse ${DEFUSE_SOURCES})
# TARGET_LINK_LIBRARIES(defuse __) 

COVISE_INSTALL_TARGET(defuse)
