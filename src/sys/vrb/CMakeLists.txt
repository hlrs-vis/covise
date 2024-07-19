# @file
# 
# Simply descend to subdirectories
#
# @author Blasius Czink

if (NOT COVISE_USE_QT)
    return()
endif()

ADD_SUBDIRECTORY(tuiProxy)
ADD_SUBDIRECTORY(vrbProxy)
ADD_SUBDIRECTORY(gui)
