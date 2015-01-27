# @file
# 
# CMakeLists.txt for covise config
#
# @author Blasius Czink

ADD_SUBDIRECTORY(kernel)
IF(COVISE_BUILD)
   ADD_SUBDIRECTORY(coEditor)
ENDIF(COVISE_BUILD)
