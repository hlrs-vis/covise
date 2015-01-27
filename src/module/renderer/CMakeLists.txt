# @file
# 
# Simply descend to subdirectories
#
# @author Blasius Czink

ADD_SUBDIRECTORY(VRMLRenderer)
IF(UNIX)
  ADD_SUBDIRECTORY(IvRenderer)
ENDIF(UNIX)

ADD_SUBDIRECTORY(WebGL)
ADD_SUBDIRECTORY(QtRenderer)
ADD_SUBDIRECTORY(Plot)
