# @file
# 
# Simply descend to subdirectories
#
# @author Blasius Czink

if(APPLE)
   return()
endif()

include_directories(../../..)

ADD_SUBDIRECTORY(lib)
ADD_SUBDIRECTORY(art)
ADD_SUBDIRECTORY(art_dtrack)
ADD_SUBDIRECTORY(cereal)
ADD_SUBDIRECTORY(fob)

# TODO: handle the rest (see above for examples)
# ADD_SUBDIRECTORY(mike)
# ADD_SUBDIRECTORY(motionstar)
# ADD_SUBDIRECTORY(polhemus)
# ADD_SUBDIRECTORY(wanda)
# ADD_SUBDIRECTORY(hornet)
# ADD_SUBDIRECTORY(vicon)
# ADD_SUBDIRECTORY(ps2)
